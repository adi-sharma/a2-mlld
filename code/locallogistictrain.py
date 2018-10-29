from utils import *
#################################

# print('Reading train data')
# # train_data, train_labels_data = read_data(train_path)
# train_data, train_labels_data = read_data(train_path, lines = 10)
# print('Reading test data')
# # test_data, test_labels_data = read_data(test_path)
# test_data, test_labels_data = read_data(test_path, lines = 10)
# # print('Reading dev data')
# # dev_data, dev_labels_data = read_data(dev_path)

loss_save = []
traintime_save = []
testtime_save = []
trainacc_save = []
testacc_save = []
t = 0
# pdb.set_trace()

W = np.zeros((num_classes, VOCAB_SIZE))
# pdb.set_trace()

for e in range(epochs):

    start = time.time()

    cumulative_loss = 0
    # b = np.random[len(labelindexer), 1]

    for i, (data, labels) in enumerate(zip(train_data, train_labels_data)):
        # print(i)
        # data = data.as_in_context(model_ctx).reshape((-1,784))
        # label = label.as_in_context(model_ctx)

        # label_one_hot = np.one_hot(label, 10)
        for label in labels:
            label_one_hot = np.zeros((num_classes))
            label_one_hot[label] +=1
            
            output = model(data, W)
            ce = cross_entropy(output, label_one_hot)
            loss = ce
            
            if i%100 ==1 and use_reg == False:
                print("epoch : %s, processed: %s, cross_entropy: %s" % (e, i, round(cumulative_loss/i, 4)))

            if i%100 ==1 and use_reg == True:

                l2 = l2reg(W, mu)
                loss -= l2
                print("epoch : %s, processed: %s, cross_entropy: %s, l2reg: %s" % (e, i, round(cumulative_loss/i, 4), round(l2, 4)))

            
            update = sgd(data, label_one_hot, output, W, lr, mu, t)
            W = (W - update)

            cumulative_loss += loss
            # pdb.set_trace()
            t += 1

    end = time.time()
    train_time = end - start
    print("Training Time = %s" % (end - start))


    test_start = time.time()
    test_accuracy = evaluate_accuracy(test_data, model, W, test_labels_data)
    test_end = time.time()
    test_time = test_end - test_start

    print("Test Time = %s" % (test_time))
    train_accuracy = evaluate_accuracy(train_data, model, W, train_labels_data)
    # dev_accuracy = evaluate_accuracy(dev_data, model, W, dev_label_data)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/len(train_data), train_accuracy, test_accuracy))

    loss_save.append(cumulative_loss/len(train_data))
    traintime_save.append(train_time)
    testtime_save.append(test_time)
    trainacc_save.append(train_accuracy)
    testacc_save.append(test_accuracy)

    def save_file(path, data):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        # pdb.set_trace()
        pickle.dump(data, open(path, 'wb'))

    file_prefix = 'epoch_'+str(e)+'_mu_'+str(mu)+'_lr_'+str(lr)+'use_sparse'+str(use_sparse)+'use_reg'+str(use_reg)

    if e%5 == 1:
        save_file(file_folder+'weights/weights_'+file_prefix, W )
        save_file(file_folder+'losses/losses_'+file_prefix, loss_save )
        save_file(file_folder+'train_time/train_time_'+file_prefix, traintime_save )
        save_file(file_folder+'test_time/test_time_'+file_prefix, testtime_save )
        save_file(file_folder+'train_accuracy/train_accuracy_'+file_prefix, trainacc_save )
        save_file(file_folder+'test_accuracy/test_accuracy_'+file_prefix, testacc_save )




pdb.set_trace()
from tempfile import TemporaryFile
outfile = TemporaryFile()

fig = plt.figure()
matplotlib.pyplot.plot(range(epochs), loss_save, '-')
fig.savefig('temp.png')

