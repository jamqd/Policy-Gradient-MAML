from policy_gradient import *
import os

def main(envs, hiddens, train_iters=50, eval_iters=5, num_workers=1):
    # directory for storing models and performancce data
    if not os.path.exists("./models"):
        os.mkdir("./models")

    if not os.path.exists("./performance_data"):
        os.mkdir("./performance_data")

    
    # run vpg experiments

    for env in envs:
        print("\nUsing environment " + env)
        if not os.path.exists("./models/" + env):
            os.mkdir("./models/" + env)
        
        if not os.path.exists("./performance_data/" + env):
            os.mkdir("./performance_data/" + env)

        for hidden_dims in hiddens:
            if not os.path.exists("./models/" + env + "/" + "_".join([str(n) for n in hidden_dims])):
                os.mkdir("./models/" + env + "/" + "_".join([str(n) for n in hidden_dims]))
        
            if not os.path.exists("./performance_data/" + env + "/" + "_".join([str(n) for n in hidden_dims])):
                os.mkdir("./performance_data/" + env + "/" + "_".join([str(n) for n in hidden_dims]))


            print("\nUsing hidden dims ", hidden_dims)

            # print("\nRunning MAML")
            # maml_pg(env_name=env, num_iterations=train_iters, policy_hidden=hidden_dims, num_workers=num_workers)

            # print("\nRunning Pretrain")
            # pretrain_pg(env_name=env, num_iterations=train_iters, policy_hidden=hidden_dims, num_workers=num_workers)
            
            # evaluate scratch vs pretrain vs maml
            print("\nTraining from scratch")
            train_pg(env_name=env, num_iterations=eval_iters, policy_hidden=hidden_dims, mode_str="scratch", num_workers=num_workers)

            print("\nTraining from MAML")
            maml_path = './models/' + env + '/' + "_".join([str(n) for n in hidden_dims]) + '/pg_maml' + '.pt'
            train_pg(env_name=env, num_iterations=eval_iters, policy_hidden=hidden_dims, filepath=maml_path, mode_str="maml", num_workers=num_workers)

            print("\nTraining from Pretrain")
            pretrain_path = './models/' + env + '/' + "_".join([str(n) for n in hidden_dims]) + '/pg_pretrain' + '.pt'
            train_pg(env_name=env, num_iterations=eval_iters, policy_hidden=hidden_dims, filepath=pretrain_path,mode_str="pretrain", num_workers=num_workers)


  

    

if __name__ == '__main__':
    # all envs have continuous action spaces
    envs = [
        'HalfCheetahForwardBackward-v1',
        'AntForwardBackward-v1',
        'HumanoidForwardBackward-v1'

    ]

    # model architectures
    hiddens = [
        [128],
        [128,128],
        [128,128,128]
    ]

    main(envs, hiddens, train_iters=50, eval_iters=3, num_workers=2)