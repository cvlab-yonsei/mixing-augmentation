import os
# import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
# from logger.logger import setup_logging
from utils.utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules,
        checkpoint saving and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log.
                       Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        self.run_id = run_id

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        # exper_name = self.config['name']
        # exper_name = self.config['data_loader']['args']['dataset'] + '_' + self.config['arch']['type'].lower()
        exper_name = f"bs_{self.config['data_loader']['args']['batch_size']}_ep_{self.config['trainer']['epochs']}"

        self.run_id = (None if self.config['name'] == "" else self.config['name'])

        if self.run_id is None:  # use timestamp as default run-id
            self.run_id = datetime.now().strftime(r'%m%d_%H%M%S')

        self._save_dir = save_dir / 'models' / exper_name / self.run_id
        self._log_dir = save_dir / 'log' / exper_name / self.run_id
        # self._save_dir = save_dir / 'models' / self.run_id
        # self._log_dir = save_dir / 'log' / self.run_id

        # make directory for saving checkpoints and log.
        exist_ok = self.run_id == ''
        try:
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        except FileExistsError:
            if self.config['test'] is True:
                run_id = datetime.now().strftime(r'%m%d_%H%M%S')
                self._save_dir = save_dir / 'models' / exper_name / f'test_{run_id}'
                # self._save_dir = save_dir / 'models' / f'test_{run_id}'
                self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
                pass
            else:
                raise FileExistsError
        try:
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
        except FileExistsError:
            if self.config['test'] is True:
                run_id = datetime.now().strftime(r'%m%d_%H%M%S')
                self._log_dir = save_dir / 'log' / exper_name / f'test_{run_id}'
                # self._log_dir = save_dir / 'log' / f'test_{run_id}'
                self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
                pass
            else:
                raise FileExistsError

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')
        
    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            if opt.action in ['store_true', 'store_false']:
                args.add_argument(*opt.flags, action=opt.action)
            else:
                args.add_argument(*opt.flags, default=None, type=opt.type, action=opt.action)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}

        if args.device is not None:
            modification['n_gpu'] = len(args.device.split(','))

        # Call constructor
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)

        try:
            return getattr(module, module_name)(*args, **module_args)
        except AttributeError:
            return None

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
