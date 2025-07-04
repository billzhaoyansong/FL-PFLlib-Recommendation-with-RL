import time
import numpy as np
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clientreputation import clientReputation


class FedReputation(Server):
    """Federated learning server with reputation management for clients.
    
    This server maintains reputation scores for all clients based on their
    contribution quality, reliability, and performance. It uses reputation
    for client selection and weighted aggregation.
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # Reputation system parameters
        self.reputation_weights = {}  # client_id -> reputation score
        self.client_contributions = {}  # client_id -> contribution history
        self.client_reliability = {}  # client_id -> reliability metrics
        self.client_performance = {}  # client_id -> performance history
        
        # Reputation parameters
        self.reputation_decay = getattr(args, 'reputation_decay', 0.95)
        self.min_reputation = getattr(args, 'min_reputation', 0.1)
        self.max_reputation = getattr(args, 'max_reputation', 2.0)
        self.reputation_threshold = getattr(args, 'reputation_threshold', 0.5)
        self.use_reputation_selection = getattr(args, 'use_reputation_selection', True)
        self.reputation_aggregation = getattr(args, 'reputation_aggregation', True)
        
        # Initialize reputation tracking
        for i in range(self.num_clients):
            self.reputation_weights[i] = 1.0  # Start with neutral reputation
            self.client_contributions[i] = []
            self.client_reliability[i] = {'participated': 0, 'total_rounds': 0, 'failures': 0}
            self.client_performance[i] = {'accuracies': [], 'losses': []}
        
        # Select slow clients and set up reputation-aware clients
        self.set_slow_clients()
        self.set_clients(clientReputation)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating reputation-aware server and clients.")
        
        self.round_time_costs = []

    def select_clients(self):
        """Select clients based on reputation scores and availability.
        
        Uses reputation-weighted selection to favor reliable and high-performing clients
        while still maintaining some randomness for fairness.
        
        Returns:
            list: Selected client objects for the round
        """
        if not self.use_reputation_selection:
            return super().select_clients()
        
        # Determine number of clients to select
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        
        # Get reputation weights for all clients
        client_indices = list(range(self.num_clients))
        reputation_scores = [self.reputation_weights[i] for i in client_indices]
        
        # Filter out clients below reputation threshold
        eligible_clients = []
        eligible_weights = []
        for i, score in zip(client_indices, reputation_scores):
            if score >= self.reputation_threshold:
                eligible_clients.append(i)
                eligible_weights.append(score)
        
        # If not enough eligible clients, include some below threshold
        if len(eligible_clients) < self.current_num_join_clients:
            remaining_clients = [i for i in client_indices if i not in eligible_clients]
            remaining_scores = [self.reputation_weights[i] for i in remaining_clients]
            
            # Add clients with highest scores among the remaining
            sorted_remaining = sorted(zip(remaining_clients, remaining_scores), 
                                    key=lambda x: x[1], reverse=True)
            needed = self.current_num_join_clients - len(eligible_clients)
            
            for client_id, score in sorted_remaining[:needed]:
                eligible_clients.append(client_id)
                eligible_weights.append(score)
        
        # Normalize weights for probability distribution
        if eligible_weights:
            total_weight = sum(eligible_weights)
            probabilities = [w / total_weight for w in eligible_weights]
        else:
            probabilities = [1.0 / len(eligible_clients)] * len(eligible_clients)
        
        # Select clients based on reputation-weighted probabilities
        try:
            selected_indices = np.random.choice(
                eligible_clients, 
                size=min(self.current_num_join_clients, len(eligible_clients)),
                replace=False, 
                p=probabilities
            )
        except ValueError:
            # Fallback to uniform selection if probability distribution is invalid
            selected_indices = np.random.choice(
                eligible_clients,
                size=min(self.current_num_join_clients, len(eligible_clients)),
                replace=False
            )
        
        selected_clients = [self.clients[i] for i in selected_indices]
        
        print(f"Selected {len(selected_clients)} clients with reputation scores: "
              f"{[self.reputation_weights[client.id] for client in selected_clients]}")
        
        return selected_clients

    def aggregate_parameters(self):
        """Aggregate client parameters using reputation-weighted averaging.
        
        Combines traditional sample-size weighting with reputation scores
        to give more influence to trustworthy clients.
        """
        assert (len(self.uploaded_models) > 0)

        self.global_model = torch.nn.utils.parameters_to_vector(self.uploaded_models[0].parameters()).clone()
        self.global_model.zero_()
        
        total_weight = 0
        
        for i, (client_model, sample_weight, client_id) in enumerate(
            zip(self.uploaded_models, self.uploaded_weights, self.uploaded_ids)):
            
            if self.reputation_aggregation:
                # Combine sample weight with reputation score
                reputation_score = self.reputation_weights[client_id]
                combined_weight = sample_weight * reputation_score
            else:
                combined_weight = sample_weight
            
            model_params = torch.nn.utils.parameters_to_vector(client_model.parameters())
            self.global_model += model_params * combined_weight
            total_weight += combined_weight
        
        # Normalize by total weight
        if total_weight > 0:
            self.global_model /= total_weight
        
        # Update the global model parameters
        torch.nn.utils.vector_to_parameters(self.global_model, self.global_model.parameters())

    def update_client_reputation(self, client_id, contribution_quality, accuracy, loss, participated=True):
        """Update reputation score for a specific client.
        
        Args:
            client_id (int): ID of the client
            contribution_quality (float): Quality of the client's contribution
            accuracy (float): Client's test accuracy
            loss (float): Client's training loss
            participated (bool): Whether client participated in the round
        """
        # Update reliability metrics
        self.client_reliability[client_id]['total_rounds'] += 1
        if participated:
            self.client_reliability[client_id]['participated'] += 1
            self.client_reliability[client_id]['failures'] = 0  # Reset failure count
        else:
            self.client_reliability[client_id]['failures'] += 1
        
        # Update performance history
        if participated:
            self.client_performance[client_id]['accuracies'].append(accuracy)
            self.client_performance[client_id]['losses'].append(loss)
            self.client_contributions[client_id].append(contribution_quality)
            
            # Keep only recent history (last 10 rounds)
            for key in ['accuracies', 'losses']:
                if len(self.client_performance[client_id][key]) > 10:
                    self.client_performance[client_id][key].pop(0)
            
            if len(self.client_contributions[client_id]) > 10:
                self.client_contributions[client_id].pop(0)
        
        # Calculate new reputation score
        reliability_score = self._calculate_reliability_score(client_id)
        performance_score = self._calculate_performance_score(client_id)
        contribution_score = self._calculate_contribution_score(client_id)
        
        # Combine scores (weighted average)
        new_reputation = (0.4 * reliability_score + 
                         0.3 * performance_score + 
                         0.3 * contribution_score)
        
        # Apply decay to current reputation and update
        current_reputation = self.reputation_weights[client_id]
        self.reputation_weights[client_id] = (self.reputation_decay * current_reputation + 
                                            (1 - self.reputation_decay) * new_reputation)
        
        # Apply bounds
        self.reputation_weights[client_id] = max(self.min_reputation, 
                                               min(self.max_reputation, 
                                                   self.reputation_weights[client_id]))

    def _calculate_reliability_score(self, client_id):
        """Calculate reliability score based on participation consistency."""
        metrics = self.client_reliability[client_id]
        if metrics['total_rounds'] == 0:
            return 1.0
        
        participation_rate = metrics['participated'] / metrics['total_rounds']
        failure_penalty = max(0, 1.0 - 0.1 * metrics['failures'])
        
        return participation_rate * failure_penalty

    def _calculate_performance_score(self, client_id):
        """Calculate performance score based on accuracy and loss trends."""
        perf = self.client_performance[client_id]
        
        if not perf['accuracies']:
            return 1.0
        
        # Recent performance (last 3 rounds vs overall average)
        recent_acc = np.mean(perf['accuracies'][-3:]) if len(perf['accuracies']) >= 3 else np.mean(perf['accuracies'])
        overall_acc = np.mean(perf['accuracies'])
        
        if overall_acc > 0:
            acc_ratio = recent_acc / overall_acc
        else:
            acc_ratio = 1.0
        
        # Normalize to [0.5, 1.5] range
        return max(0.5, min(1.5, acc_ratio))

    def _calculate_contribution_score(self, client_id):
        """Calculate contribution score based on training improvements."""
        contributions = self.client_contributions[client_id]
        
        if not contributions:
            return 1.0
        
        avg_contribution = np.mean(contributions)
        
        # Normalize contribution (assuming contribution is loss improvement)
        if avg_contribution > 0:
            return min(1.5, 1.0 + avg_contribution)
        else:
            return max(0.5, 1.0 + avg_contribution)

    def train(self):
        """Main training loop with reputation tracking."""
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # Track which clients were selected for training
            selected_client_ids = [client.id for client in self.selected_clients]
            
            # Train selected clients
            for client in self.selected_clients:
                client.train()

            # Update reputation for all clients
            self._update_all_reputations(selected_client_ids)

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.round_time_costs.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.round_time_costs[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.round_time_costs[1:]) / len(self.round_time_costs[1:]))
        
        # Print final reputation scores
        self._print_reputation_summary()

        self.save_results()
        self.save_global_model()

    def _update_all_reputations(self, selected_client_ids):
        """Update reputation for all clients based on their participation and performance."""
        for client_id in range(self.num_clients):
            participated = client_id in selected_client_ids
            
            if participated:
                client = self.clients[client_id]
                # Get client's performance metrics
                test_acc, test_num, auc = client.test_metrics()
                train_loss, train_num = client.train_metrics()
                
                accuracy = test_acc / test_num if test_num > 0 else 0
                loss = train_loss / train_num if train_num > 0 else float('inf')
                
                # Calculate contribution quality (could be loss improvement or other metrics)
                contribution_quality = max(0, 1.0 - loss)  # Simple metric based on normalized loss
                
                self.update_client_reputation(client_id, contribution_quality, accuracy, loss, True)
            else:
                # Client didn't participate - mark as non-participation
                self.update_client_reputation(client_id, 0, 0, float('inf'), False)

    def _print_reputation_summary(self):
        """Print summary of client reputations."""
        print("\n" + "="*50)
        print("CLIENT REPUTATION SUMMARY")
        print("="*50)
        
        for client_id in range(self.num_clients):
            reputation = self.reputation_weights[client_id]
            reliability = self.client_reliability[client_id]
            participation_rate = (reliability['participated'] / max(1, reliability['total_rounds'])) * 100
            
            print(f"Client {client_id:2d}: Reputation={reputation:.3f}, "
                  f"Participation={participation_rate:.1f}%, "
                  f"Failures={reliability['failures']}")
        
        print("="*50)

    def get_reputation_weights(self):
        """Get current reputation weights for all clients.
        
        Returns:
            dict: Mapping from client_id to reputation score
        """
        return self.reputation_weights.copy()

    def set_reputation_weights(self, reputation_dict):
        """Set reputation weights for clients.
        
        Args:
            reputation_dict (dict): Mapping from client_id to reputation score
        """
        for client_id, score in reputation_dict.items():
            if client_id in self.reputation_weights:
                self.reputation_weights[client_id] = max(self.min_reputation,
                                                       min(self.max_reputation, score))