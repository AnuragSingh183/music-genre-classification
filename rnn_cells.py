import numpy as np


class RNNCell:
    def __init__(self, input_size, hidden_size):
        """Initialize a basic RNN cell."""
        # Xavier/Glorot initialization for weights
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
        self.bh = np.zeros((hidden_size, 1))
        
        # For output
        self.Why = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (2 * hidden_size))
        self.by = np.zeros((hidden_size, 1))
        
        # Memory for the backward pass
        self.h_cache = []
        self.x_cache = []
        
    def forward(self, x, h_prev):
        """Forward pass of RNN cell."""
        # Combine input and previous hidden state
        h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
        
        # Save for backward pass
        self.h_cache.append(h)
        self.x_cache.append(x)
        
        return h
    
    def backward(self, dh_next, learn_rate=0.001, max_norm=5):
        """Backward pass of RNN cell with Gradient Clipping."""
        dh = dh_next.copy()
        h = self.h_cache.pop()
        x = self.x_cache.pop()
        h_prev = self.h_cache[-1] if self.h_cache else np.zeros_like(h)
        
        # Backprop through tanh
        dtanh = (1 - h**2) * dh

        # Gradients for the weights
        dWxh = np.dot(dtanh, x.T)
        dWhh = np.dot(dtanh, h_prev.T)
        dbh = np.sum(dtanh, axis=1, keepdims=True)

        # 🔥 Gradient Clipping for stability
        np.clip(dWxh, -max_norm, max_norm, out=dWxh)
        np.clip(dWhh, -max_norm, max_norm, out=dWhh)
        np.clip(dbh, -max_norm, max_norm, out=dbh)

        # Gradients for the next backward step
        dx = np.dot(self.Wxh.T, dtanh)
        dh_prev = np.dot(self.Whh.T, dtanh)

        # Update weights
        self.Wxh -= learn_rate * dWxh
        self.Whh -= learn_rate * dWhh
        self.bh -= learn_rate * dbh  # Corrected broadcast

        return dx, dh_prev

    
    def reset_memory(self):
        """Reset the memory caches."""
        self.h_cache = []
        self.x_cache = []


class GRUCell:
    def __init__(self, input_size, hidden_size):
        """Initialize a GRU cell."""
        # Update gate
        self.Wz = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.Uz = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (2 * hidden_size))
        self.bz = np.zeros((hidden_size, 1))
        
        # Reset gate
        self.Wr = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.Ur = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (2 * hidden_size))
        self.br = np.zeros((hidden_size, 1))
        
        # Candidate hidden state
        self.Wh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.Uh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (2 * hidden_size))
        self.bh = np.zeros((hidden_size, 1))
        
        # Memory for the backward pass
        self.cache = []
        
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def forward(self, x, h_prev):
        """Forward pass of GRU cell."""
        # Update gate
        z = self.sigmoid(np.dot(self.Wz, x) + np.dot(self.Uz, h_prev) + self.bz)
        
        # Reset gate
        r = self.sigmoid(np.dot(self.Wr, x) + np.dot(self.Ur, h_prev) + self.br)
        
        # Candidate hidden state
        h_candidate = np.tanh(np.dot(self.Wh, x) + np.dot(self.Uh, r * h_prev) + self.bh)
        
        # New hidden state
        h = z * h_prev + (1 - z) * h_candidate
        
        # Save for backward pass
        self.cache.append((x, h_prev, z, r, h_candidate, h))
        
        return h
        
    def backward(self, dh_next, learn_rate=0.001):
        """Backward pass of GRU cell."""
        x, h_prev, z, r, h_candidate, h = self.cache.pop()

        dh = dh_next.copy()

        # Backprop through the update equation
        dh_candidate = dh * (1 - z)
        dh_prev_from_h = dh * z
        dz = dh * (h_prev - h_candidate)

        # Backprop through the candidate hidden state
        dtanh = dh_candidate * (1 - h_candidate**2)
        dUh = np.dot(dtanh, (r * h_prev).T)
        dr_from_h_candidate = np.dot(self.Uh.T, dtanh) * h_prev
        dWh = np.dot(dtanh, x.T)
        dx_from_h_candidate = np.dot(self.Wh.T, dtanh)
        dbh = np.sum(dtanh, axis=1, keepdims=True)  # Fixed shape

        # Backprop through the reset gate
        dr = dr_from_h_candidate
        dsigmoid_r = dr * r * (1 - r)
        dUr = np.dot(dsigmoid_r, h_prev.T)
        dWr = np.dot(dsigmoid_r, x.T)
        dx_from_r = np.dot(self.Wr.T, dsigmoid_r)
        dh_prev_from_r = np.dot(self.Ur.T, dsigmoid_r)
        dbr = np.sum(dsigmoid_r, axis=1, keepdims=True)  # Fixed shape

        # Backprop through the update gate
        dsigmoid_z = dz * z * (1 - z)
        dUz = np.dot(dsigmoid_z, h_prev.T)
        dWz = np.dot(dsigmoid_z, x.T)
        dx_from_z = np.dot(self.Wz.T, dsigmoid_z)
        dh_prev_from_z = np.dot(self.Uz.T, dsigmoid_z)
        dbz = np.sum(dsigmoid_z, axis=1, keepdims=True)  # Fixed shape

        # Combine gradients
        dx = dx_from_h_candidate + dx_from_r + dx_from_z
        dh_prev = dh_prev_from_h + dh_prev_from_r + dh_prev_from_z

        # Update weights
        self.Wz -= learn_rate * dWz
        self.Uz -= learn_rate * dUz
        self.bz -= learn_rate * dbz  # Corrected shape

        self.Wr -= learn_rate * dWr
        self.Ur -= learn_rate * dUr
        self.br -= learn_rate * dbr

        self.Wh -= learn_rate * dWh
        self.Uh -= learn_rate * dUh
        self.bh -= learn_rate * dbh

        return dx, dh_prev

    def reset_memory(self):
        """Reset the memory caches."""
        self.cache = []


class BiRNN:
    def __init__(self, input_size, hidden_size):
        """Initialize a Bidirectional RNN."""
        self.forward_rnn = RNNCell(input_size, hidden_size // 2)  # Split hidden size for forward and backward
        self.backward_rnn = RNNCell(input_size, hidden_size // 2)
        self.hidden_size = hidden_size
        
    def forward(self, X):
        """Forward pass through the BiRNN."""
        batch_size, seq_len, feature_dim = X.shape
        
        # Initialize hidden states
        h_forward = np.zeros((batch_size, self.hidden_size // 2))
        h_backward = np.zeros((batch_size, self.hidden_size // 2))
        
        # Forward direction
        forward_states = []
        for t in range(seq_len):
            h_forward = self.forward_rnn.forward(X[:, t, :].T, h_forward.T).T
            forward_states.append(h_forward)
        
        # Backward direction
        backward_states = []
        for t in range(seq_len - 1, -1, -1):
            h_backward = self.backward_rnn.forward(X[:, t, :].T, h_backward.T).T
            backward_states.insert(0, h_backward)  # Insert at beginning to maintain order
        
        # Concatenate forward and backward states
        combined_states = []
        for t in range(seq_len):
            combined_states.append(np.concatenate((forward_states[t], backward_states[t]), axis=1))
        
        return combined_states

class RNNModel:
    def __init__(self, input_size, hidden_size, output_size, cell_type='vanilla'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell_type = cell_type

        if cell_type == 'vanilla':
            self.rnn_cell = RNNCell(input_size, hidden_size)
        elif cell_type == 'gru':
            self.rnn_cell = GRUCell(input_size, hidden_size)
        elif cell_type == 'birnn':
            self.rnn_cell = BiRNN(input_size, hidden_size)
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")
        
        # Added Dense Layer for better feature mapping
        self.dense = np.random.randn(hidden_size // 2, hidden_size) * np.sqrt(2.0 / (hidden_size // 2 + hidden_size))

        self.Why = np.random.randn(output_size, hidden_size // 2) * np.sqrt(2.0 / (output_size + hidden_size // 2))

        # self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / (output_size + hidden_size))
        self.by = np.zeros((output_size, 1))

        self.y_cache = []
        self.softmax_cache = []

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, X):
        batch_size, seq_len, feature_dim = X.shape
        
        if self.cell_type == 'birnn':
            hidden_states = self.rnn_cell.forward(X)
            final_h = hidden_states[-1].T
        else:
            h = np.zeros((self.hidden_size, batch_size))
            
            for t in range(seq_len):
                x_t = X[:, t, :].T
                h = self.rnn_cell.forward(x_t, h)

            final_h = h

        # 🔥 Correct Dense Layer Application
        dense_out = np.tanh(np.dot(self.dense, final_h))

        y = np.dot(self.Why, dense_out) + self.by
        self.y_cache.append((dense_out, y))

        probs = self.softmax(y)
        self.softmax_cache.append(probs)

        return probs

    def backward(self, y_true, learn_rate=0.001):
        """Backward for RNN Model with Gradient Clipping."""
        final_h, y = self.y_cache.pop()
        probs = self.softmax_cache.pop()

        dprobs = probs.copy()
        y_true_onehot = np.zeros_like(dprobs)
        for i, label in enumerate(y_true):
            y_true_onehot[label, i] = 1

        dprobs -= y_true_onehot
        dWhy = np.dot(dprobs, final_h.T)
        dby = np.sum(dprobs, axis=1, keepdims=True)

        # Backpropagate through Dense Layer
        dh = np.dot(self.Why.T, dprobs)

        # 🔥 Map Dense output back to original hidden size
        dh = np.dot(self.dense.T, dh)  # Mapping back to 128 features

        # Update weights
        self.Why -= learn_rate * dWhy
        self.by -= learn_rate * dby

        if self.cell_type != 'birnn':
            dh_next = dh
            dx, _ = self.rnn_cell.backward(dh_next, learn_rate)


    def train(self, X, y, epochs=10, batch_size=32, learn_rate=0.001):
        n_samples = X.shape[0]
        losses = []
        accuracies = []

        for epoch in range(epochs):
            if hasattr(self.rnn_cell, 'reset_memory'):
                self.rnn_cell.reset_memory()

            total_loss = 0
            correct_predictions = 0

            # 🔥 Learning Rate Decay Implementation
            current_lr = learn_rate / (1 + 0.01 * epoch)  # Decay over epochs

            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                if hasattr(self.rnn_cell, 'reset_memory'):
                    self.rnn_cell.reset_memory()

                probs = self.forward(X_batch)

                batch_size_actual = y_batch.shape[0]
                y_indices = np.array(range(batch_size_actual))
                loss = -np.sum(np.log(probs[y_batch, y_indices] + 1e-10)) / batch_size_actual
                total_loss += loss

                predictions = np.argmax(probs, axis=0)
                correct_predictions += np.sum(predictions == y_batch)

                # 🔥 Use the decayed learning rate in the backward pass
                self.backward(y_batch, current_lr)

            avg_loss = total_loss / (n_samples / batch_size)
            accuracy = correct_predictions / n_samples

            losses.append(avg_loss)
            accuracies.append(accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, LR: {current_lr:.5f}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return losses, accuracies


    def predict(self, X):
        if hasattr(self.rnn_cell, 'reset_memory'):
            self.rnn_cell.reset_memory()

        probs = self.forward(X)
        return np.argmax(probs, axis=0)