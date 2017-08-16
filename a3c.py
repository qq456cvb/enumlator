from emulator import Emulator
import os, random
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn

PASS_PENALTY = 5


def update_params(scope_from, scope_to):
    vars_from = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_from)
    vars_to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_to)

    ops = []
    for from_var, to_var in zip(vars_from, vars_to):
        ops.append(to_var.assign(from_var))
    return ops


def discounted_return(r, gamma):
    r = r.astype(float)
    r_out = np.zeros_like(r)
    val = 0
    for i in reversed(range(r.shape[0])):
        r_out[i] = r[i] + gamma * val
        val = r_out[i]
    return r_out


class CardNetwork:
    def __init__(self, s_dim, trainer, scope, a_dim=8310):
        with tf.variable_scope(scope):
            self.input = tf.placeholder(tf.float32, [None, s_dim], name="input")
            self.fc1 = slim.fully_connected(inputs=self.input, num_outputs=256)
            self.fc2 = slim.fully_connected(inputs=self.fc1, num_outputs=128)

            self.lstm = rnn.BasicLSTMCell(num_units=256, state_is_tuple=True)

            # batch size 1 for every step
            c_input = tf.placeholder(tf.float32, [1, self.lstm.state_size.c])
            h_input = tf.placeholder(tf.float32, [1, self.lstm.state_size.h])

            step_cnt = tf.shape(self.input)[:1]
            self.rnn_hidden_in = rnn.LSTMStateTuple(c_input, h_input)
            rnn_in = tf.expand_dims(self.fc2, 0)
            rnn_out, self.rnn_hidden_out = tf.nn.dynamic_rnn(self.lstm, rnn_in,
                                                             initial_state=self.rnn_hidden_in,
                                                             sequence_length=step_cnt)
            # lstm_c_output, lstm_h_output = lstm_state_output
            self.rnn_out = tf.reshape(rnn_out, [-1, 256])

            self.fc3 = slim.fully_connected(inputs=self.rnn_out, num_outputs=1024)
            self.policy_pred = slim.fully_connected(inputs=self.fc3, num_outputs=a_dim, activation_fn=None)

            self.mask = tf.placeholder(tf.bool, [None, a_dim], name='mask')
            self.mask = tf.reshape(self.mask, [1, -1])
            self.valid_policy = tf.boolean_mask(self.policy_pred[0], self.mask[0])
            self.valid_policy = tf.nn.softmax(self.valid_policy)
            self.valid_policy = tf.reshape(self.valid_policy, [1, -1])

            self.val_output = slim.fully_connected(inputs=self.rnn_out, num_outputs=1, activation_fn=None)
            self.val_pred = tf.reshape(self.val_output, [-1])

            # only support batch size one since masked_a_dim is changing
            self.action = tf.placeholder(tf.int32, [None], "action_input")
            self.policy_penalty = tf.placeholder(tf.float32, [None], "policy_penalty")
            self.masked_a_dim = tf.placeholder(tf.int32, None)
            self.action_one_hot = tf.one_hot(self.action, self.masked_a_dim, dtype=tf.float32)

            self.val_truth = tf.placeholder(tf.float32, [None], "val_input")
            self.advantages = tf.placeholder(tf.float32, [None], "advantage_input")

            self.pi_sample = tf.reduce_sum(self.action_one_hot * self.valid_policy, [1])
            self.policy_loss = -tf.reduce_sum(self.policy_penalty * tf.log(tf.clip_by_value(self.pi_sample, 1e-10, 1.)) * self.advantages)

            self.val_loss = tf.reduce_sum(tf.square(self.val_pred-self.val_truth))

            self.loss = 0.4 * self.val_loss + self.policy_loss

            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.gradients = tf.gradients(self.loss, local_vars)

            self.gradients, _ = tf.clip_by_global_norm(self.gradients, 40.0)
            # global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
            self.apply_grads = trainer.apply_gradients(zip(self.gradients, local_vars))


class CardAgent:
    def __init__(self, name, trainer):
        self.name = name
        self.episodes = tf.Variable(0, dtype=tf.int32, name='episodes_' + name, trainable=False)
        self.increment = self.episodes.assign_add(1)
        self.network = CardNetwork(54 * 5, trainer, self.name, 8310)

    def train_batch(self, rnn_backup, buffer, masks, sess, gamma, val_last):
        states = buffer[:, 0]
        actions = buffer[:, 1]
        rewards = buffer[:, 2]
        values = buffer[:, 3]
        a_dims = buffer[0, 4]
        policy_penalty = buffer[:, 5]

        rewards_plus = np.append(rewards, val_last)
        val_truth = discounted_return(rewards_plus, gamma)[:-1]

        val_pred_plus = np.append(values, val_last)
        td0 = rewards + gamma * val_pred_plus[1:] - val_pred_plus[:-1]
        advantages = discounted_return(td0, gamma)

        sess.run(self.network.apply_grads, feed_dict={self.network.val_truth: val_truth,
                                                      self.network.advantages: advantages,
                                                      self.network.input: np.vstack(states),
                                                      self.network.rnn_hidden_in[0]: rnn_backup[0],
                                                      self.network.rnn_hidden_in[1]: rnn_backup[1],
                                                      self.network.action: actions,
                                                      self.network.masked_a_dim: a_dims,
                                                      self.network.mask: masks,
                                                      self.network.policy_penalty: policy_penalty})


class CardMaster:
    def __init__(self, env):
        self.name = 'global'
        self.env = env
        self.a_dim = 8310
        self.gamma = 0.99

        # DO NOT MODIFY THIS since masked action is needed
        self.train_intervals = 1

        self.trainer = tf.train.AdamOptimizer()
        self.episode_rewards = [[]] * 3
        self.episode_length = [[]] * 3
        self.episode_mean_values = [[]] * 3
        self.summary_writers = [tf.summary.FileWriter("train_agent%d" % i) for i in range(3)]

        self.agents = [CardAgent('agent%d' % i, self.trainer) for i in range(3)]

        self.global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        self.increment = self.global_episodes.assign_add(1)

    def train_batch(self, rnn_backup, buffer, masks, sess, gamma, val_last, idx):
        buffer = np.array(buffer)
        masks = np.array(masks)
        self.agents[idx].train_batch(rnn_backup, buffer, masks, sess, gamma, val_last)

    def run(self, sess, saver, max_episode_length):
        with sess.as_default():
            global_episodes = sess.run(self.global_episodes)
            while global_episodes < 10001:
                print("episode %d" % global_episodes)
                episode_buffer = []
                episode_mask = []
                episode_values = []
                episode_reward = 0
                episode_steps = 0

                self.env.begin()
                self.env.prepare()

                train_id = self.env.id - 1
                print("training id %d" % train_id)
                s = self.env.get_state()
                s = np.reshape(s, [1, -1])

                # init LSTM state
                c_init = np.zeros((1, self.agents[train_id].network.lstm.state_size.c), np.float32)
                h_init = np.zeros((1, self.agents[train_id].network.lstm.state_size.h), np.float32)
                rnn_state = [c_init, h_init]
                rnn_state_backup = rnn_state

                for l in range(max_episode_length):
                    print("turn %d" % l)
                    mask = self.env.get_mask()
                    oppo_cnt = self.env.get_opponent_min_cnt()
                    # encourage response when opponent is about to win
                    if random.random() > oppo_cnt / 20. and np.count_nonzero(mask) > 1:
                        mask[0] = False
                    policy, val, rnn_state = sess.run([self.agents[train_id].network.valid_policy,
                                                       self.agents[train_id].network.val_pred,
                                                      self.agents[train_id].network.rnn_hidden_out],
                                                      feed_dict={self.agents[train_id].network.input: s,
                                                                 self.agents[train_id].network.mask: np.reshape(mask, [1, -1]),
                                                                 self.agents[train_id].network.rnn_hidden_in[0]: rnn_state[0],
                                                                 self.agents[train_id].network.rnn_hidden_in[1]: rnn_state[1]
                                                                 })
                    policy = policy[0]
                    valid_actions = np.take(np.arange(self.a_dim), mask.nonzero())
                    valid_actions = valid_actions.reshape(-1)
                    # valid_p = np.take(policy[0], mask.nonzero())
                    # valid_p = valid_p / np.sum(valid_p)
                    # valid_p = valid_p.reshape(-1)
                    # print(policy.shape)
                    # print(valid_actions.shape)
                    a = np.random.choice(valid_actions, p=policy)
                    a_masked = np.where(valid_actions == a)[0]

                    r, done = self.env.step(a)
                    s_prime = self.env.get_state()
                    s_prime = np.reshape(s_prime, [1, -1])

                    episode_buffer.append([s, a_masked, r, val[0], np.sum(mask.astype(np.float32)), PASS_PENALTY if a == 0 else 1])
                    episode_mask.append(mask)
                    episode_values.append(val)
                    episode_reward += r
                    episode_steps += 1

                    if done:
                        print(np.sum(self.env.history))
                        if len(episode_buffer) != 0:
                            self.train_batch(rnn_state_backup, episode_buffer, episode_mask, sess, self.gamma, 0, train_id)
                        break

                    s = s_prime

                    if len(episode_buffer) == self.train_intervals:
                        val_last = sess.run(self.agents[train_id].network.val_pred,
                                            feed_dict={self.agents[train_id].network.input: s,
                                                       self.agents[train_id].network.rnn_hidden_in[0]: rnn_state[0],
                                                       self.agents[train_id].network.rnn_hidden_in[1]: rnn_state[1]
                                                       })
                        self.train_batch(rnn_state_backup, episode_buffer, episode_mask, sess, self.gamma, val_last[0], train_id)
                        episode_buffer = []
                        episode_mask = []

                        rnn_state_backup = rnn_state

                self.episode_mean_values[train_id].append(np.mean(episode_values))
                self.episode_length[train_id].append(episode_steps)
                self.episode_rewards[train_id].append(episode_reward)

                episodes = sess.run(self.agents[train_id].episodes)
                sess.run(self.agents[train_id].increment)

                global_episodes += 1
                sess.run(self.increment)
                if global_episodes % 50 == 0:
                    saver.save(sess, './model' + '/model-' + str(global_episodes) + '.cptk')
                    print("Saved Model")

                if episodes % 5 == 0 and episodes > 0:
                    mean_reward = np.mean(self.episode_rewards[train_id][-5:])
                    mean_length = np.mean(self.episode_length[train_id][-5:])
                    mean_value = np.mean(self.episode_mean_values[train_id][-5:])

                    summary = tf.Summary()
                    summary.value.add(tag='rewards', simple_value=float(mean_reward))
                    summary.value.add(tag='length', simple_value=float(mean_length))
                    summary.value.add(tag='values', simple_value=float(mean_value))

                    self.summary_writers[train_id].add_summary(summary, episodes)
                    self.summary_writers[train_id].flush()

                self.env.end()


def run_game(sess, network):
    max_episode_length = 100
    lord_win_rate = 0
    for i in range(100):
        network.env.reset()
        network.env.players[0].trainable = True
        lord_idx = 2
        network.env.players[2].is_human = True
        network.env.prepare(lord_idx)

        s = network.env.get_state(0)
        s = np.reshape(s, [1, -1])

        while True:
            policy, val = sess.run([network.agent.network.policy_pred, network.agent.network.val_pred],
                                   feed_dict={network.agent.network.input: s})
            mask = network.env.get_mask(0)
            valid_actions = np.take(np.arange(network.a_dim), mask.nonzero())
            valid_actions = valid_actions.reshape(-1)
            valid_p = np.take(policy[0], mask.nonzero())
            if np.count_nonzero(valid_p) == 0:
                valid_p = np.ones([valid_p.size]) / float(valid_p.size)
            else:
                valid_p = valid_p / np.sum(valid_p)
            valid_p = valid_p.reshape(-1)
            a = np.random.choice(valid_actions, p=valid_p)

            r, done = network.env.step(0, a)
            s_prime = network.env.get_state(0)
            s_prime = np.reshape(s_prime, [1, -1])

            if done:
                idx = network.env.check_winner()
                if idx == lord_idx:
                    lord_win_rate += 1
                print("winner is player %d" % idx)
                print("..............................")
                break
            s = s_prime
    print("lord winning rate: %f" % (lord_win_rate / 100.0))

if __name__ == '__main__':
    # x = tf.placeholder(tf.float32, [3])
    # m = tf.placeholder(tf.bool, [3])
    #
    # a = tf.placeholder(tf.int32, [None])
    # a_onehot = tf.one_hot(a, 3, dtype=tf.float32)
    # valid_x = tf.boolean_mask(x, m)
    # valid_x = tf.nn.softmax(valid_x)
    # y = tf.Variable([0., 0., 0.], trainable=False)
    # y[1:3].assign(valid_x)
    # grad = tf.gradients(y[1], x)
    #
    # with tf.Session() as sess:
    #     mask = np.zeros([3])
    #     mask[2] = 1
    #     mask[1] = 1
    #     mask = mask.astype(np.bool)
    #     g, x = sess.run([grad, valid_x],
    #                     feed_dict={x: np.ones([3]) * 2,
    #                             m: mask})
    #     # print(y)
    #     print(g)
    #     print(x)

    load_model = False
    model_path = './model'
    cardgame = Emulator()
    with tf.device("/gpu:0"):
        master = CardMaster(cardgame)
    saver = tf.train.Saver(max_to_keep=20)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            master.run(sess, saver, 2000)
            # run_game(sess, master)
        else:
            sess.run(tf.global_variables_initializer())
            master.run(sess, saver, 2000)
        sess.close()
