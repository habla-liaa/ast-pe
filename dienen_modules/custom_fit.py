import tensorflow as tf
import tqdm

def custom_fit(data,model,epochs=None,callbacks=None,initial_epoch=0, loss=None, strategy=None):
    optimizer = model.optimizer
    if loss is None:
        loss_obj = model.loss
    else:
        loss_obj = loss

    def loss_fn(model,loss_data):
        x,y = loss_data
        y_pred = model(x,training=True)

        return loss_obj(y, y_pred)

    for cb in callbacks: cb.model = model
    for cb in callbacks: cb.on_train_begin({})
    for epoch in range(initial_epoch,epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        model.reset_metrics()
        for cb in callbacks: cb.on_epoch_begin(epoch)

        step_counter = tqdm.tqdm(range(len(data)))
        for step in step_counter:
            try:
                for cb in callbacks: cb.on_train_batch_begin(step)
                loss_data = data.__getitem__(step)
                with tf.GradientTape() as tape:
                    loss_value = loss_fn(model, loss_data)
                step_logs = {m.name: m.result() for m in model.metrics}
                step_logs['batch_loss'] = loss_value
                epoch_loss_avg.update_state(loss_value)
                step_logs['loss'] = epoch_loss_avg.result()
                step_counter.set_description('Epoch {}/{} Batch loss: {:.3e} Mean loss: {:.3e}'.format(epoch+1,epochs,loss_value,step_logs['loss']))
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                for cb in callbacks: cb.on_train_batch_end(step,logs=step_logs)
            except Exception as e:
                print(e)
        val_logs={'loss': epoch_loss_avg.result()}
        for cb in callbacks: cb.on_epoch_end(epoch,logs=val_logs)
        data.on_epoch_end()
    for cb in callbacks: cb.on_train_end()