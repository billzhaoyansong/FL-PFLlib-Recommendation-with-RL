import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientReputation(Client):
    """Federated learning client with reputation tracking capabilities.
    
    This client maintains reputation scores for other known clients and itself,
    tracks contribution quality, and provides mechanisms for reputation-based
    collaboration and evaluation.
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Initialize reputation tracking for all known clients
        self.num_known_clients = getattr(args, 'num_clients', 100)
        self.client_reputations = {}  # client_id -> reputation score
        self.client_interactions = {}  # client_id -> interaction history
        self.client_performance_history = {}  # client_id -> performance metrics
        
        # Initialize reputation scores for all clients (including self)
        for client_id in range(self.num_known_clients):
            self.client_reputations[client_id] = 1.0  # Start with neutral reputation
            self.client_interactions[client_id] = []
            self.client_performance_history[client_id] = {
                'accuracies': [],
                'losses': [],
                'contributions': []
            }
        
        # Self-reputation tracking
        self.reputation_score = 1.0  # Own reputation score
        self.contribution_history = []  # Track own contribution quality over time
        self.accuracy_history = []  # Track local model accuracy
        self.reliability_score = 1.0  # Based on consistency of participation
        self.data_quality_score = 1.0  # Based on data quality assessment
        
        # Reputation parameters
        self.reputation_decay = kwargs.get('reputation_decay', 0.95)
        self.min_reputation = kwargs.get('min_reputation', 0.1)
        self.max_reputation = kwargs.get('max_reputation', 2.0)
        self.trust_threshold = kwargs.get('trust_threshold', 0.5)
        
        # Participation tracking
        self.total_rounds = 0
        self.participated_rounds = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = kwargs.get('max_consecutive_failures', 3)
        
        # Collaborative learning parameters
        self.enable_reputation_filtering = kwargs.get('enable_reputation_filtering', True)
        self.reputation_weighted_learning = kwargs.get('reputation_weighted_learning', False)

    def train(self):
        """Enhanced training with reputation tracking and quality assessment."""
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()
        initial_loss = self.get_current_loss()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # Calculate training improvement for reputation
        final_loss = self.get_current_loss()
        loss_improvement = max(0, initial_loss - final_loss)
        self.update_own_contribution_score(loss_improvement)
        
        # Update participation tracking
        self.participated_rounds += 1
        self.consecutive_failures = 0  # Reset on successful training
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def get_current_loss(self):
        """Calculate current training loss for reputation assessment."""
        trainloader = self.load_train_data()
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                total_loss += loss.item() * y.shape[0]
                total_samples += y.shape[0]
        
        return total_loss / total_samples if total_samples > 0 else 0

    def update_own_contribution_score(self, improvement):
        """Update own contribution score based on training improvement."""
        self.contribution_history.append(improvement)
        
        # Keep only recent history (last 10 rounds)
        if len(self.contribution_history) > 10:
            self.contribution_history.pop(0)
        
        # Calculate average contribution
        avg_contribution = np.mean(self.contribution_history) if self.contribution_history else 0
        
        # Update reputation based on contribution
        contribution_factor = min(2.0, max(0.5, 1.0 + avg_contribution))
        self.reputation_score *= contribution_factor
        
        # Apply bounds
        self.reputation_score = max(self.min_reputation, 
                                   min(self.max_reputation, self.reputation_score))

    def update_peer_reputation(self, client_id, accuracy, loss, contribution_quality):
        """Update reputation score for another client based on observed performance.
        
        Args:
            client_id (int): ID of the client to update
            accuracy (float): Observed accuracy of the client
            loss (float): Observed loss of the client
            contribution_quality (float): Quality of client's contribution
        """
        if client_id not in self.client_reputations:
            self.client_reputations[client_id] = 1.0
            self.client_interactions[client_id] = []
            self.client_performance_history[client_id] = {
                'accuracies': [], 'losses': [], 'contributions': []
            }
        
        # Record interaction
        interaction = {
            'round': self.total_rounds,
            'accuracy': accuracy,
            'loss': loss,
            'contribution': contribution_quality
        }
        self.client_interactions[client_id].append(interaction)
        
        # Update performance history
        history = self.client_performance_history[client_id]
        history['accuracies'].append(accuracy)
        history['losses'].append(loss)
        history['contributions'].append(contribution_quality)
        
        # Keep only recent history (last 10 interactions)
        for key in history:
            if len(history[key]) > 10:
                history[key].pop(0)
        
        # Calculate new reputation based on recent performance
        recent_accuracy = np.mean(history['accuracies'][-3:]) if len(history['accuracies']) >= 3 else np.mean(history['accuracies'])
        recent_contribution = np.mean(history['contributions'][-3:]) if len(history['contributions']) >= 3 else np.mean(history['contributions'])
        
        # Performance factor (higher accuracy and contribution = better reputation)
        performance_factor = min(1.5, max(0.5, recent_accuracy + recent_contribution))
        
        # Update reputation with decay
        current_rep = self.client_reputations[client_id]
        self.client_reputations[client_id] = (self.reputation_decay * current_rep + 
                                            (1 - self.reputation_decay) * performance_factor)
        
        # Apply bounds
        self.client_reputations[client_id] = max(self.min_reputation,
                                               min(self.max_reputation,
                                                   self.client_reputations[client_id]))

    def get_trusted_clients(self):
        """Get list of client IDs that are considered trustworthy.
        
        Returns:
            list: Client IDs with reputation above trust threshold
        """
        trusted = []
        for client_id, reputation in self.client_reputations.items():
            if reputation >= self.trust_threshold and client_id != self.id:
                trusted.append(client_id)
        return trusted

    def filter_clients_by_reputation(self, client_list):
        """Filter a list of clients based on their reputation scores.
        
        Args:
            client_list (list): List of client IDs to filter
            
        Returns:
            list: Filtered list of trustworthy clients
        """
        if not self.enable_reputation_filtering:
            return client_list
        
        filtered = []
        for client_id in client_list:
            if client_id in self.client_reputations:
                if self.client_reputations[client_id] >= self.trust_threshold:
                    filtered.append(client_id)
            else:
                # Unknown clients get neutral treatment
                filtered.append(client_id)
        
        return filtered

    def get_reputation_weighted_average(self, client_models, client_ids):
        """Calculate reputation-weighted average of client models.
        
        Args:
            client_models (list): List of model parameters
            client_ids (list): List of corresponding client IDs
            
        Returns:
            torch.Tensor: Weighted average model parameters
        """
        if not self.reputation_weighted_learning or not client_models:
            # Fall back to simple average
            return torch.stack(client_models).mean(dim=0)
        
        weights = []
        for client_id in client_ids:
            if client_id in self.client_reputations:
                weights.append(self.client_reputations[client_id])
            else:
                weights.append(1.0)  # Neutral weight for unknown clients
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Calculate weighted average
        weighted_model = torch.zeros_like(client_models[0])
        for model, weight in zip(client_models, weights):
            weighted_model += model * weight
        
        return weighted_model

    def update_accuracy_reputation(self, test_accuracy):
        """Update own reputation based on test accuracy."""
        self.accuracy_history.append(test_accuracy)
        
        # Keep only recent history (last 10 rounds)
        if len(self.accuracy_history) > 10:
            self.accuracy_history.pop(0)
        
        # Calculate accuracy-based reputation factor
        if len(self.accuracy_history) >= 2:
            recent_accuracy = np.mean(self.accuracy_history[-3:])  # Last 3 rounds
            overall_accuracy = np.mean(self.accuracy_history)
            
            # Reward improving accuracy
            if recent_accuracy > overall_accuracy:
                accuracy_factor = 1.1
            elif recent_accuracy < overall_accuracy * 0.9:
                accuracy_factor = 0.9
            else:
                accuracy_factor = 1.0
            
            self.reputation_score *= accuracy_factor
            self.reputation_score = max(self.min_reputation, 
                                       min(self.max_reputation, self.reputation_score))

    def update_reliability_score(self):
        """Update reliability score based on participation consistency."""
        self.total_rounds += 1
        
        if self.total_rounds > 0:
            participation_rate = self.participated_rounds / self.total_rounds
            self.reliability_score = participation_rate
            
            # Penalize consecutive failures
            if self.consecutive_failures > 0:
                failure_penalty = 0.9 ** self.consecutive_failures
                self.reliability_score *= failure_penalty

    def report_failure(self):
        """Report a failure in participation or training."""
        self.consecutive_failures += 1
        self.total_rounds += 1
        
        # Apply penalty for failures
        failure_penalty = 0.8 ** self.consecutive_failures
        self.reputation_score *= failure_penalty
        
        # Apply bounds
        self.reputation_score = max(self.min_reputation, 
                                   min(self.max_reputation, self.reputation_score))
        
        self.update_reliability_score()

    def get_reputation_metrics(self):
        """Get current reputation metrics for this client.
        
        Returns:
            dict: Comprehensive reputation metrics
        """
        return {
            'own_reputation': self.reputation_score,
            'reliability_score': self.reliability_score,
            'data_quality_score': self.data_quality_score,
            'participation_rate': self.participated_rounds / max(1, self.total_rounds),
            'consecutive_failures': self.consecutive_failures,
            'avg_contribution': np.mean(self.contribution_history) if self.contribution_history else 0,
            'avg_accuracy': np.mean(self.accuracy_history) if self.accuracy_history else 0,
            'trusted_clients': len(self.get_trusted_clients()),
            'total_known_clients': len(self.client_reputations)
        }

    def get_peer_reputations(self):
        """Get reputation scores for all known clients.
        
        Returns:
            dict: Mapping from client_id to reputation score
        """
        return self.client_reputations.copy()

    def set_peer_reputation(self, client_id, reputation_score):
        """Set reputation score for a specific client.
        
        Args:
            client_id (int): ID of the client
            reputation_score (float): New reputation score
        """
        bounded_score = max(self.min_reputation, 
                           min(self.max_reputation, reputation_score))
        self.client_reputations[client_id] = bounded_score

    def is_reliable(self):
        """Check if this client is considered reliable for participation."""
        return (self.reputation_score >= self.min_reputation and 
                self.consecutive_failures < self.max_consecutive_failures)

    def get_weighted_contribution(self):
        """Get reputation-weighted contribution for aggregation."""
        return self.reputation_score

    def decay_reputations(self):
        """Apply time-based decay to all reputation scores."""
        # Decay own reputation
        self.reputation_score *= self.reputation_decay
        self.reputation_score = max(self.min_reputation, self.reputation_score)
        
        # Decay peer reputations
        for client_id in self.client_reputations:
            self.client_reputations[client_id] *= self.reputation_decay
            self.client_reputations[client_id] = max(self.min_reputation, 
                                                   self.client_reputations[client_id])

    def test_metrics(self):
        """Override test_metrics to include reputation updates."""
        test_acc, test_num, auc = super().test_metrics()
        
        # Update own reputation based on test accuracy
        if test_num > 0:
            accuracy = test_acc / test_num
            self.update_accuracy_reputation(accuracy)
        
        return test_acc, test_num, auc

    def share_reputation_info(self, other_client_reputations):
        """Share and receive reputation information with other clients.
        
        Args:
            other_client_reputations (dict): Reputation scores from another client
        """
        # Simple reputation sharing: average with other client's scores
        sharing_weight = 0.3  # How much to trust other client's opinions
        
        for client_id, other_reputation in other_client_reputations.items():
            if client_id in self.client_reputations:
                current_rep = self.client_reputations[client_id]
                # Weighted average of own opinion and other's opinion
                self.client_reputations[client_id] = (
                    (1 - sharing_weight) * current_rep + 
                    sharing_weight * other_reputation
                )
            else:
                # New client - adopt other's opinion with reduced confidence
                self.client_reputations[client_id] = sharing_weight * other_reputation + (1 - sharing_weight) * 1.0