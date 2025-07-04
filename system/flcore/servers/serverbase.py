import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
from abc import ABC, abstractmethod


class Server(ABC):
    def __init__(self, args, times):
        """Initialize the federated learning server.
        
        Args:
            args: Configuration arguments containing model, dataset, and training parameters
            times: Number of experimental runs for statistical analysis
        """
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.selected_clients_per_round = []
        self.train_slow_mask = []
        self.send_slow_mask = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        """Initialize and set up all clients for federated learning.
        
        Args:
            clientObj: Client class to instantiate for each client
        """
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_mask, self.send_slow_mask):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        """Randomly selects a proportion of clients to be designated as "slow".

        This method simulates client heterogeneity by marking a random subset of
        clients as "slow". These clients can then have artificial delays
        introduced during training or communication to model real-world stragglers.

        Args:
            slow_rate (float): The proportion of clients to mark as slow (0.0 to 1.0).

        Returns:
            list[bool]: A list of booleans of length `self.num_clients`.
                        `True` at an index indicates that the client is slow.
        """
        num_slow_clients = int(slow_rate * self.num_clients)
        slow_indices = np.random.choice(range(self.num_clients), num_slow_clients, replace=False)
        slow_clients_mask = [False] * self.num_clients
        for i in slow_indices:
            slow_clients_mask[i] = True
        return slow_clients_mask

    def set_slow_clients(self):
        """Initialize slow client masks for training and communication delays.
        
        This method sets up masks to simulate client heterogeneity by designating
        some clients as slow for training and/or communication.
        """
        self.train_slow_mask = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_mask = self.select_slow_clients(
            self.send_slow_rate)

    @abstractmethod
    def train(self):
        """Main training loop for the server.
        This method should be implemented by all subclasses."""
        raise NotImplementedError

    def select_clients(self):
        """Selects a subset of clients for the current training round.

        This method first determines the number of clients to join the round.
        If `self.random_join_ratio` is True, it selects a random number of
        clients between `self.num_join_clients` and `self.num_clients`.
        Otherwise, it uses the fixed `self.num_join_clients`.

        It then randomly samples the determined number of clients from the
        full client list without replacement.

        Returns:
            list: A list of the selected client objects for the round.
        """
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        # Record selected client IDs for this round
        self.selected_clients_per_round.append([c.id for c in selected_clients])
        return selected_clients

    def send_models(self):
        """Send the global model to all clients.
        
        Distributes the current global model parameters to all clients and
        tracks communication time costs for each client.
        """
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        """Receive and collect model updates from active clients.
        
        Collects models from clients that participated in training, accounting
        for client dropout and time constraints. Normalizes client weights based
        on training sample sizes.
        """
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        """Aggregate client model parameters to update the global model.
        
        Performs weighted averaging of client models based on their training
        sample sizes to create the new global model.
        """
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        """Add weighted client parameters to the global model.
        
        Args:
            w (float): Weight for the client model (typically based on sample size)
            client_model: Client model whose parameters to add
        """
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        """Save the current global model to disk.
        
        Saves the global model in the models directory with algorithm-specific naming.
        """
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        """Load a previously saved global model from disk.
        
        Loads the global model from the models directory based on dataset and algorithm.
        """
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        """Check if a saved global model exists on disk.
        
        Returns:
            bool: True if the model file exists, False otherwise
        """
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        """Save experimental results to HDF5 file.
        
        Saves test accuracy, test AUC, and training loss results for analysis.
        """
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        """Save a general item to the server's save folder.
        
        Args:
            item: Object to save (typically a PyTorch tensor or model)
            item_name (str): Name identifier for the saved item
        """
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        """Load a previously saved item from the server's save folder.
        
        Args:
            item_name (str): Name identifier of the item to load
            
        Returns:
            Loaded item object
        """
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        """Collect test metrics from all clients.
        
        Evaluates all clients on their test data and collects accuracy and AUC metrics.
        If new clients are being evaluated, performs fine-tuning first.
        
        Returns:
            tuple: (client_ids, num_samples, total_correct, total_auc)
        """
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        """Collect training metrics from all clients.
        
        Evaluates all clients on their training data and collects loss metrics.
        
        Returns:
            tuple: (client_ids, num_samples, losses)
        """
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        """Evaluate the global model performance across all clients.
        
        Computes and prints averaged test accuracy, test AUC, and training loss.
        Optionally appends results to provided lists for tracking.
        
        Args:
            acc (list, optional): List to append test accuracy results
            loss (list, optional): List to append training loss results
        """
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        if acc == None:
            self.rs_test_auc.append(test_auc)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        """Print formatted evaluation metrics.
        
        Args:
            test_acc (float): Test accuracy value
            test_auc (float): Test AUC value
            train_loss (float): Training loss value
        """
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        """Check if training convergence criteria are met.
        
        Evaluates convergence based on top performance count and/or standard deviation.
        
        Args:
            acc_lss: List of accuracy lists to check
            top_cnt (int, optional): Minimum rounds since best performance
            div_value (float, optional): Maximum standard deviation threshold
            
        Returns:
            bool: True if convergence criteria are met, False otherwise
        """
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        """Perform Deep Leakage from Gradients (DLG) attack evaluation.
        
        Evaluates privacy vulnerability by attempting to reconstruct training data
        from model gradients using the DLG attack method.
        
        Args:
            R (int): Current round number for logging purposes
        """
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        """Initialize new clients for evaluation purposes.
        
        Creates additional clients beyond the training set to evaluate
        generalization to unseen participants.
        
        Args:
            clientObj: Client class to instantiate for each new client
        """
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        """Fine-tune the global model on new clients.
        
        Performs local fine-tuning on new clients using the current global model
        to evaluate adaptation to new participants.
        """
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        """Collect test metrics from new clients.
        
        Evaluates new clients on their test data after fine-tuning.
        
        Returns:
            tuple: (client_ids, num_samples, total_correct, total_auc)
        """
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
