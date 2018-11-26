import helper
import re
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np
# 导入seq2seq，下面会用他计算loss
from tensorflow.contrib import seq2seq


dir = './data/斗破苍穹.txt'
text = helper.load_text(dir)
num_words_for_training = 30000

text = text[:num_words_for_training]
lines_of_text = text.split('\n')

print(len(lines_of_text))
print(lines_of_text[:5])

lines_of_text = [lines for lines in lines_of_text if len(lines) > 0]

print(len(lines_of_text))
print(lines_of_text[:5])

# 去掉每行首尾空格
lines_of_text = [lines.strip() for lines in lines_of_text]
print(lines_of_text[:5])

# 生成一个正则，负责找『[]』包含的内容
pattern = re.compile(r'\[.*\]')
# 将所有指定内容替换成空
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
print(lines_of_text[:5])

# 将上面的正则换成负责找『<>』包含的内容
pattern = re.compile(r'<.*>')
# 将所有指定内容替换成空
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]

# 将上面的正则换成负责找『......』包含的内容
pattern = re.compile(r'\.+')
# 将所有指定内容替换成空
lines_of_text = [pattern.sub("。", lines) for lines in lines_of_text]

# 将上面的正则换成负责找行中的空格
pattern = re.compile(r' +')
# 将所有指定内容替换成空
lines_of_text = [pattern.sub("，", lines) for lines in lines_of_text]

# 将上面的正则换成负责找句尾『\\r』的内容
pattern = re.compile(r'\\r')
# 将所有指定内容替换成空
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
print(lines_of_text[:5])
#########################################################################################################

#因为模型只认识数字，不认识中文，所以将文字对应到数字，分别创建文字:数字和数字:文字的两个字典
def create_lookup_tables(input_data):
    vocab = set(input_data)

    # 文字到数字的映射
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}

    # 数字到文字的映射
    int_to_vocab = dict(enumerate(vocab))

    return vocab_to_int, int_to_vocab


def token_lookup():
    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n'])

    tokens = ["P", "C", "Q", "T", "S", "E", "M", "I", "O", "D", "R"]

    return dict(zip(symbols, tokens))


helper.preprocess_and_save_data(''.join(lines_of_text), token_lookup, create_lookup_tables)
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



# 训练循环次数
# 训练循环次数
num_epochs = 200

# batch大小
batch_size = 256

# lstm层中包含的unit个数
rnn_size = 512

# embedding layer的大小
embed_dim = 512

# 训练步长
seq_length = 30

# 学习率
learning_rate = 0.003

# 每多少步打印一次训练信息
show_every_n_batches = 30

# 保存session状态的位置
save_dir = './save'

# 创建输入，目标以及学习率的placeholder
# tf.placeholder(dtype, shape=None, name=None)
# 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
#
# 参数：
#
# dtype：数据类型。常用的是tf.float32,tf.float64等数值类型shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定name：名称。
# 返回：Tensor 类型
def get_inputs():
    # inputs和targets的类型都是整数的
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return inputs, targets, learning_rate

#创建rnn cell，使用lstm cell，并创建相应层数的lstm层，应用dropout，以及初始化lstm层状态。
def get_init_cell(batch_size, rnn_size):
    # lstm层数
    num_layers = 2

    # dropout时的保留概率
    keep_prob = 0.8

    # 创建包含rnn_size个神经元的lstm cell
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    # 使用dropout机制防止overfitting等
    drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    # 创建2层lstm层
    cell = tf.contrib.rnn.MultiRNNCell([drop for _ in range(num_layers)])

    # 初始化状态为0.0
    init_state = cell.zero_state(batch_size, tf.float32)

    # 使用tf.identify给init_state取个名字，后面生成文字的时候，要使用这个名字来找到缓存的state
    init_state = tf.identity(init_state, name='init_state')

    return cell, init_state


def get_embed(input_data, vocab_size, embed_dim):
    # 先根据文字数量和embedding layer的size创建tensorflow variable
    embedding = tf.Variable(tf.truncated_normal([vocab_size, embed_dim], stddev=0.1),
                            dtype=tf.float32, name="embedding")

    # 让tensorflow帮我们创建lookup table
    return tf.nn.embedding_lookup(embedding, input_data, name="embed_data")


def build_rnn(cell, inputs):
    '''
    cell就是上面get_init_cell创建的cell
    '''

    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    # 同样给final_state一个名字，后面要重新获取缓存
    final_state = tf.identity(final_state, name="final_state")

    return outputs, final_state


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    # 创建embedding layer
    embed = get_embed(input_data, vocab_size, embed_dim)

    # 计算outputs 和 final_state
    outputs, final_state = build_rnn(cell, embed)

    # remember to initialize weights and biases, or the loss will stuck at a very high point
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                               biases_initializer=tf.zeros_initializer())

    # logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)

    return logits, final_state


def get_batches(int_text, batch_size, seq_length):
    # # 计算有多少个batch可以创建
    # n_batches = (len(int_text) // (batch_size * seq_length))
    #
    # # 计算每一步的原始数据，和位移一位之后的数据
    # batch_origin = np.array(int_text[: n_batches * batch_size * seq_length])
    # batch_shifted = np.array(int_text[1: n_batches * batch_size * seq_length + 1])
    #
    # # 将位移之后的数据的最后一位，设置成原始数据的第一位，相当于在做循环
    # batch_shifted[-1] = batch_origin[0]
    #
    # batch_origin_reshape = np.split(batch_origin.reshape(batch_size, -1), n_batches, 1)
    # batch_shifted_reshape = np.split(batch_shifted.reshape(batch_size, -1), n_batches, 1)
    #
    # batches = np.array(list(zip(batch_origin_reshape, batch_shifted_reshape)))

    characters_per_batch = batch_size * seq_length
    num_batches = len(int_text) // characters_per_batch

    # clip arrays to ensure we have complete batches for inputs, targets same but moved one unit over
    input_data = np.array(int_text[: num_batches * characters_per_batch])
    target_data = np.array(int_text[1: num_batches * characters_per_batch + 1])

    inputs = input_data.reshape(batch_size, -1)
    targets = target_data.reshape(batch_size, -1)

    inputs = np.split(inputs, num_batches, 1)
    targets = np.split(targets, num_batches, 1)

    batches = np.array(list(zip(inputs, targets)))
    batches[-1][-1][-1][-1] = batches[0][0][0][0]

    return batches

train_graph = tf.Graph()
with train_graph.as_default():
    # 文字总量
    vocab_size = len(int_to_vocab)

    # 获取模型的输入，目标以及学习率节点，这些都是tf的placeholder
    input_text, targets, lr = get_inputs()

    # 输入数据的shape
    input_data_shape = tf.shape(input_text)

    # 创建rnn的cell和初始状态节点，rnn的cell已经包含了lstm，dropout
    # 这里的rnn_size表示每个lstm cell中包含了多少的神经元
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)

    # 创建计算loss和finalstate的节点
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # 使用softmax计算最后的预测概率
    probs = tf.nn.softmax(logits, name='probs')

    # 计算loss
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # 使用Adam提督下降
    optimizer = tf.train.AdamOptimizer(lr)

    # 裁剪一下Gradient输出，最后的gradient都在[-1, 1]的范围内
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

# 获得训练用的所有batch
batches = get_batches(int_text, batch_size, seq_length)

# 打开session开始训练，将上面创建的graph对象传递给session
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # 打印训练信息
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')

# 将使用到的变量保存起来，以便下次直接读取。
helper.save_params((seq_length, save_dir))


