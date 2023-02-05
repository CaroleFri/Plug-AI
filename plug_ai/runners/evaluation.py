import torch

@torch.no_grad()
def eval_loop(self):
    """
    Run the evaluation on the whole validation dataset and return the expected metric.

    Arguments:
        self (object): Trainer object containing at least a validation dataset, a MONAI metric, a model and an inference method.
        
    Returns:
        mean_score (Tensor): aggregate of all scores in the metric 
    """
    self.model.train()
    self.metric.reset()
    total_eval_step = len(self.val_loader)
    for i, sample in enumerate(self.val_loader):
        pred = self.inference_step(sample["input"])
        pred_shape = pred.shape

        # choose the class with the highest probability and reshape the tensor for the loss
        target = torch.nn.functional.one_hot(
            torch.argmax(pred, 1),
            # get number of class from output channel
            num_classes=pred_shape[1]
        )
        # reshape for computing metric
        target = target.view(pred_shape).cpu()

        # Better on cpu to avoid cuda out of memory
        eval_score = self.metric(target, sample["label"]) # In case we need every score

        print(f'[Step_Eval {i+1}/{total_eval_step}]')

        # if self.report_log:
        #     self.writer.add_scalar('Loss/train', loss.item(), i + epoch * total_train_step)

    print(f"Evaluation score for this epoch: {self.metric.aggregate()}")

    return self.metric.aggregate()
