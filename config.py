class params:
    def __init__(self): 
        
        self.cat = "lamp"
        
        #DGCNN parameters
        self.emb_dims = 256
        self.dropout = 0.5
        self.k = 20 
        
        #EdgePooling parameters
        self.k_pool1 = 16
        self.k_pool2 = 6
        self.pool1_points = 512
        self.pool2_points = 128
        self.scoring_fun = "tanh"

        #Decoder parameters 
        self.num_branch = 8
        self.K1 = 64
        self.K2 = 64
        self.N = 128
        self.method = "integrated"
        self.alpha = 0.01


        #Multi-head Attention parameters
        self.d_attn = 256
        self.num_heads = 4

        #Training parameters
        self.batch_size = 32
        self.nThreads = 8
        self.lr = 0.0002
        self.dataroot = "../ShapeNetViPC/ShapeNetViPC-Dataset"
        self.n_epochs = 200
        self.ckp_dir = "./checkpoints_path"
        self.ckp_epoch = 5
        self.eval_epoch = 5
        self.resume = ''
        self.loss_print = 500
        self.vis_step = 1

        self.stage = 1
        

