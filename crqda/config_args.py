class Args:
    def __init__(self):
        self.model_type = 'roberta'
        self.model_name_or_path = '/data/model/'
        self.output_dir = '/data/output'
        self.data_dir = None
        self.train_file = '/data/squad/train-v2.0.json'
        self.predict_file = '/data/squad/dev-v2.0.json'
        self.config_name = ""
        self.tokenizer_name=''
        self.cache_dir=""

        self.version_2_with_negative=True
        self.null_score_diff_threshold=0.0
                        

        self.max_seq_length = 384
        self.doc_stride=128
        self.max_query_length=64
        self.do_train = False
        self.do_eval= False
        self.evaluate_during_training=False
        self.do_lower_case=False

        self.per_gpu_train_batch_size = 6
        self.per_gpu_eval_batch_size = 12
        self.learning_rate=5e-5
        self.gradient_accumulation_steps=1
        self.weight_decay=0.01
        self.adam_epsilon=1e-8
        self.max_grad_norm=1.0
        self.num_train_epochs=100
        self.max_steps=-1
        self.warmup_steps=500
        self.n_best_size=20
        self.max_answer_length=30
        self.verbose_logging=False

        self.logging_steps=50
        self.save_steps=5000
        self.eval_all_checkpoints=False
        self.no_cuda=False
        self.overwrite_output_dir=False
        self.overwrite_cache=True
        self.seed=555

        self.local_rank=-1
        self.fp16=False
        self.fp16_opt_level='O1'
                           
        self.server_ip=''
        self.server_port=''

        self.threads=24
        
