from keras.callbacks import Callback


class MyCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        epoch_file_path = 'checkpoints/epoch.txt'
        try:
            with open(epoch_file_path, 'r') as f:
                temp = int(f.read())
            with open(epoch_file_path, 'w') as f:
                f.write(str(temp + 1))
        except IOError:
            with open(epoch_file_path, 'w') as f:
                f.write(str(epoch + 1))
