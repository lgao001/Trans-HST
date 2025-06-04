from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/mnt/b2730ee6-a71b-4fb0-8470-ae63626b6f38/clk/TransT/checkpoints'    # Where tracking networks are stored.
    # settings.network_path = '/amax/GL/TransT-main-final1/checkpoints/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/mnt/b2730ee6-a71b-4fb0-8470-ae63626b6f38/clk/TransT/pytracking/result_plots/'
    settings.results_path = '/mnt/b2730ee6-a71b-4fb0-8470-ae63626b6f38/clk/TransT/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/mnt/b2730ee6-a71b-4fb0-8470-ae63626b6f38/clk/TransT/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

