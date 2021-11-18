from absl import app
from absl import logging
import os
import sys
import signal
import datetime
from absl import flags
from utils.utils import program_duration
import train_utils
import pseudo_label
FLAGS = flags.FLAGS
flags.DEFINE_string("training_type", help="train func", default="original")
flags.DEFINE_string("dl", help="dataset preparation", default="my")
flags.DEFINE_enum("alg", enum_values=["PL", "supervised"],
                  help="PseudoLabel or supervised", default="PL")


def main(argv):
    dt1 = datetime.datetime.now()
    del argv  # not used
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    # from train_utils import set_dataset, set_model, get_log_name, log_accuracy, start_training, start_self_learning
    dso, data_config = train_utils.set_dataset(FLAGS.dataset, FLAGS.lt, FLAGS.semi, scale=FLAGS.dl == "my")
    # model = train_utils.set_model(FLAGS.network, data_config, FLAGS.weights, FLAGS.lt, FLAGS.opt, FLAGS.lr)
    model, ssl_obj, optimizer, criterion = pseudo_label.set_model(FLAGS.network, data_config, FLAGS.weights, FLAGS.lt, FLAGS.opt, FLAGS.lr)
    # set up logging details
    log_dir, log_name = train_utils.get_log_name(FLAGS, data_config, prefix="pl")
    os.makedirs(log_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(log_name, log_dir)
    logging.get_absl_handler().setFormatter(None)
    # FLAGS.append_flags_into_file('config/' + FLAGS.dataset + '-' + FLAGS.network + '-' + FLAGS.lt + '-.cfg')
    print("training logs are saved at: ", log_dir)
    logging.info(FLAGS.flag_values_dict())

    # logging initial accuracy before training
    ac = train_utils.log_accuracy(model, dso, FLAGS.lt, FLAGS.semi, labelling=FLAGS.lbl)
    logging.info("init Test accuracy : {:.2f} %".format(ac))

    def ctrl_c_accuracy():  # will print accuracy if ctrl+c pressed
        ac_ = train_utils.log_accuracy(model, dso, FLAGS.lt, FLAGS.semi, labelling=FLAGS.lbl)
        logging.info("ctrl_c Test accuracy : {:.2f} %".format(ac_))
        print(program_duration(dt1, 'Killed after Time'))

    def exit_gracefully(signum, frame):   # signal handling
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, original_sigint)
        try:
            if input("\nReally quit? (y/n)> ").lower().startswith('y'):
                ctrl_c_accuracy()
                sys.exit(1)
        except KeyboardInterrupt:
            print("Ok ok, quitting")
            sys.exit(1)

    signal.signal(signal.SIGINT, exit_gracefully)

    # start training on N-labelled and log accuracy
    pseudo_label.start_training([model, ssl_obj, optimizer, criterion], dso, FLAGS.epochs, FLAGS.semi, FLAGS.batch_size
                                ,verb=FLAGS.verbose,alg=FLAGS.alg, train_type=FLAGS.training_type, dl=FLAGS.dl)
    ac = train_utils.log_accuracy(model, dso, FLAGS.lt, FLAGS.semi, labelling=FLAGS.lbl)
    logging.info("after training, Test accuracy : {:.2f} %".format(ac))
    # apply self-training
    if FLAGS.self_training:
        FLAGS.verbose = 2
        for g in optimizer.param_groups:  # reduce lr by factor of 0.1
            g['lr'] = FLAGS.lr/10.
        pseudo_label.start_self_learning([model, optimizer, criterion], dso, data_config, FLAGS.lt, FLAGS.meta_iterations,
                                        FLAGS.epochs_per_m_iteration, FLAGS.batch_size, logging)

    print(program_duration(dt1, 'Total Time taken'))


if __name__ == '__main__':
    from flags import setup_flags
    setup_flags()
    FLAGS.alsologtostderr = True  # also show logging info to std output

    app.run(main)
