#encoding=utf-8

#训练文档矩阵

#read:
# softmax_weights_init.pkl

#write:
# doc_embeddings.pkl

from utils import *
import tensorflow as tf

dictionary, _, vocab_size, data, doclens = build_dictionary()
twcp = get_text_window_center_positions(data)

np.random.shuffle(twcp)
twcp_train_gen = repeater_shuffler(twcp) #中间单词迭代器
del twcp # save some memory

def init_weights():
    glove_file = 'data/glove/glove.6B.%dd.txt' % EMBEDDING_SIZE
    weights = create_glove_embedding_init(dictionary, glove_file)
    with open('softmax_weights_init.pkl','wb') as f:
        pickle.dump(weights,f)

def create_training_graph():
    regularizer = tf.contrib.layers.l2_regularizer(0.004)
    # Input data
    dataset = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
    # document Weights
    embeddings = tf.get_variable(name='embeddings',
                                 regularizer=regularizer,
                                 initializer=tf.random_uniform([len(doclens), DOC_EMBEDDING_SIZE],
                                                               -1.0, 1.0))
    # Model
    # Look up embeddings for inputs
    embed = tf.nn.embedding_lookup(embeddings, dataset)

    emb_dense_layer1 = tf.contrib.layers.fully_connected(embed,
                                                  2*EMBEDDING_SIZE,
                                                  activation_fn=None,
                                                  weights_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02),
                                                  scope="dense_expanddim_embed")
    emb_bn_layer1 = tf.contrib.layers.batch_norm(emb_dense_layer1,
                                          center=True, scale=True,
                                          scope='embed_bn0')
    emb_repr1 = tf.nn.relu(emb_bn_layer1)
    emb_dense_layer2 = tf.contrib.layers.fully_connected(emb_repr1,
                                                           EMBEDDING_SIZE,
                                                           activation_fn=None,
                                                           weights_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02),
                                                           scope="dense_decreasedim_embed")
    emb_bn_layer2 = tf.contrib.layers.batch_norm(emb_dense_layer2,
                                          center=True, scale=True,
                                          scope='embed_bn1')
    emb_repr2 = tf.nn.relu(emb_bn_layer2)

    with open('./data/softmax_weights_init.pkl', 'rb') as file:
        weights_init = pickle.load(file)
        softmax_weights = tf.get_variable(name='softmax_weights',
                                          initializer=weights_init,
                                          regularizer=regularizer)
    softmax_biases = tf.get_variable(name='softmax_biases',
                                     regularizer=regularizer,
                                     initializer=tf.zeros([vocab_size]))

    # Compute the softmax loss, using a sample of the negative
    # labels each time
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(
            softmax_weights, softmax_biases, labels,
            emb_repr2, NUM_SAMPLED, vocab_size))
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss=loss+regularization_loss

    # Optimizer
    optimizer = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(total_loss)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    return dataset, labels, softmax_weights, softmax_biases, total_loss, optimizer, embeddings, session

def generate_batch_single_twcp(twcp, i, batch, labels):
    tw_start = twcp - (TEXT_WINDOW_SIZE - 1) // 2
    tw_end = twcp + TEXT_WINDOW_SIZE // 2 + 1
    docids, wordids = zip(*data[tw_start:tw_end])
    batch_slice = slice(i * TEXT_WINDOW_SIZE,(i + 1) * TEXT_WINDOW_SIZE)
    batch[batch_slice] = docids
    labels[batch_slice, 0] = wordids


def generate_batch(twcp_gen):
    batch = np.ndarray(shape=(BATCH_SIZE,), dtype=np.int32)
    labels = np.ndarray(shape=(BATCH_SIZE, 1), dtype=np.int32)
    for i in range(BATCH_SIZE // TEXT_WINDOW_SIZE):
        generate_batch_single_twcp(next(twcp_gen), i, batch, labels)
    return batch, labels

def train(optimizer, loss, dataset, labels):
    avg_training_loss = 0
    for step in range(NUM_STEPS):
        batch_data, batch_labels = generate_batch(twcp_train_gen)
        _, l = session.run([optimizer, loss],feed_dict={dataset: batch_data, labels: batch_labels})
        avg_training_loss += l
        if step > 0 and step % REPORT_EVERY_X_STEPS == 0:
            avg_training_loss = avg_training_loss / REPORT_EVERY_X_STEPS
            print('Average loss at step {:d}: {:.1f}'.format(step, avg_training_loss))

dataset, labels, _, _, loss, optimizer, embeddings, session = create_training_graph()
train(optimizer, loss, dataset, labels)

current_embeddings = session.run(embeddings)

with open('data/doc_embeddings.pkl','wb') as file:
    pickle.dump(current_embeddings,file)
