from lightning import Callback

class PrintMetricsCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bỏ qua bước kiểm tra nhanh (sanity check) đầu tiên
        if trainer.sanity_checking:
            return
            
        metrics = trainer.callback_metrics
        
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs
        
        # Lấy giá trị của Train (có thể không tồn tại ngay epoch 0 nên dùng .item() cẩn thận)
        train_loss = metrics.get('train/loss', 0.0)
        train_loss_seg = metrics.get('seg_train/loss', 0.0)
        train_loss_cls = metrics.get('cls_train/loss', 0.0)
        
        # Ở code cũ người ta gọi là "test_loss", nhưng thực chất trong quá trình train nó là "val_loss"
        val_loss = metrics.get('val/loss', 0.0)
        val_loss_seg = metrics.get('seg_val/loss', 0.0)
        val_loss_cls = metrics.get('cls_val/loss', 0.0)
        
        # Chuyển tensor thành float nếu cần
        if hasattr(train_loss, 'item'): train_loss = train_loss.item()
        if hasattr(train_loss_seg, 'item'): train_loss_seg = train_loss_seg.item()
        if hasattr(train_loss_cls, 'item'): train_loss_cls = train_loss_cls.item()
        if hasattr(val_loss, 'item'): val_loss = val_loss.item()
        if hasattr(val_loss_seg, 'item'): val_loss_seg = val_loss_seg.item()
        if hasattr(val_loss_cls, 'item'): val_loss_cls = val_loss_cls.item()
        
        print(f"\nepoch: {epoch}/{max_epochs}:")
        print(f"            \ttrain_loss: {train_loss:.4f}, train_loss_seg: {train_loss_seg:.4f}, train_loss_cls: {train_loss_cls:.4f},")
        print(f"            \ttest_loss: {val_loss:.4f}, test_loss_seg : {val_loss_seg:.4f}, test_loss_cls : {val_loss_cls:.4f},")
