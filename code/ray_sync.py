from utils import *
# import subprocess
# subprocess.run(['source set_node.sh'], shell = True)
ray.init(
    redis_address="10.24.28.102:8379", 
    # object_store_memory=10000000000000
    )
num_workers = 4

#####################################################################

loss_save = []
t = 0
l2=0
W = np.zeros((num_classes, VOCAB_SIZE))
# b = np.random[len(labelindexer), 1]

# @ray.remote
# class ParameterServer(object):
#     def __init__(self, learning_rate):
#         self.net = model.SimpleCNN(learning_rate=learning_rate)

#     def apply_gradients(self, *gradients):
#         self.net.apply_gradients(np.mean(gradients, axis=0))
#         return self.net.variables.get_flat()

#     def get_weights(self):
#         return self.net.variables.get_flat()

# @ray.remote
# class Worker(object):
#     def __init__(self, worker_index, batch_size=50):
#         self.worker_index = worker_index
#         self.batch_size = batch_size
#         self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True,
#                                                seed=worker_index)
#         self.net = model.SimpleCNN()

#     def compute_gradients(self, weights):
#         self.net.variables.set_flat(weights)
#         xs, ys = self.mnist.train.next_batch(self.batch_size)
# #         return self.net.compute_gradients(xs, ys)
# while True:
#     gradients = [worker.compute_gradients.remote(current_weights)
#                  for worker in workers]
#     current_weights = ps.apply_gradients.remote(*gradients)


@ray.remote
class ParameterServer(object):
    def __init__(self, W, cumulative_loss = 0, t = 0):
        self.weights = W.copy()
        self.cumulative_loss = cumulative_loss
        self.t = t
        self.epoch_num = 0
        self.processed_num = 0

    def apply_gradients(self, *gradients):
        update, cumulative_loss, t = gradients
        self.weights -= update
        self.cumulative_loss += loss
        self.t += 1

    def get_weights(self):
        return self.weights, self.cumulative_loss, self.t

    def get_params(self):
        return self.cumulative_loss, self.t, self.epoch_num, self.processed_num

    def epoch_next(self):
        self.epoch_num += 1
        self.processed_num = 0
        self.cumulative_loss = 0

    def increase_count(self):
        self.processed_num += 1

@ray.remote
def worker_task(ps):

    def compute_gradients(self, W):
        # if n < len(train_data) and e < epochs:
        loss, t, e, n = ps.get_params.remote()

        data, labels = train_data[n], train_labels_data[n]

        for label in labels:
            label_one_hot = np.zeros((num_classes))
            try:
                label_one_hot[label] +=1
            except:
                print('label not found: %s' % (label))
            output = model(data, W)

            ce = cross_entropy(output, label_one_hot)
            loss = ce
            if use_reg:
                l2 = l2reg(W, mu)
                loss += l2
            
            update = sgd(data, label_one_hot, output, W, lr, mu, t)
            
            self.increase_count()

            if n == len(train_data)-1:
                self.epoch_next()

        return update

ps = ParameterServer.remote(W, cumulative_loss=0)
dump = (([worker_task.remote(ps) for i in range(num_workers)]))


# loss_save = []
# traintime_save = []
# testtime_save = []
# trainacc_save = []
# testacc_save = []

while True:
    gradients = [worker.compute_gradients.remote(current_weights)
                 for worker in workers]
    current_weights = ps.apply_gradients.remote(*gradients)

    test_accuracy = evaluate_accuracy(test_data, model, W, test_labels_data)
    train_accuracy = evaluate_accuracy(train_data, model, W, train_labels_data)
    # dev_accuracy = evaluate_accuracy(dev_data, model, W, dev_label_data)
    print("Epoch %s, Counter : %s, Loss: %s, Train_acc %s, Test_acc %s" % (e, t, cumulative_loss/(t%len(train_data) + 1), train_accuracy, test_accuracy))
    print('\007')
    print('\007')
    print('\007')

    # loss_save.append(cumulative_loss/len(train_data))
    # traintime_save.append(train_time)
    # testtime_save.append(test_time)
    # trainacc_save.append(train_accuracy)
    # testacc_save.append(test_accuracy)
    
    file_prefix = 'epoch_'+str(e)+'_mu_'+str(mu)+'_lr_'+str(lr)+'use_sparse'+str(use_sparse)+'use_reg'+str(use_reg)

    if e%5 == 1:
        save_file('../models/ray_async/weights/weights_'+file_prefix, W )
        save_file('../models/ray_async/losses/losses_'+file_prefix, loss_save )
        save_file('../models/ray_async/train_time/train_time_'+file_prefix, traintime_save )
        save_file('../models/ray_async/test_time/test_time_'+file_prefix, testtime_save )
        save_file('../models/ray_async/train_accuracy/train_accuracy_'+file_prefix, trainacc_save )
        save_file('../models/ray_async/test_accuracy/test_accuracy_'+file_prefix, testacc_save )



from tempfile import TemporaryFile
outfile = TemporaryFile()

fig = plt.figure()
plt.plot(range(epochs), loss_save, '-')
fig.savefig('temp.png')

## Saving
# export OMP_NUM_THREADS=1