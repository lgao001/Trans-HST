class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/b2730ee6-a71b-4fb0-8470-ae63626b6f38/clk/TransT/checkpoints'    # Base directory for saving network checkpoints.
        # self.workspace_dir = '/amax/GL/TransT-main-final1/checkpoints'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = ''
        self.got10k_dir = '/mnt/b2730ee6-a71b-4fb0-8470-ae63626b6f38/clk/HSI-lpp/HSI16/train'
        # self.got10k_dir = '/mnt/b2730ee6-a71b-4fb0-8470-ae63626b6f38/clk/RGB/got-10k/train'
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
