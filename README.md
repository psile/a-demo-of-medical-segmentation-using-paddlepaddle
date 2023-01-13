# deeplearning/medical segmentation

first solve medical segmentation problem using paddlepaddle
大二时用paddlepaddle做的医学图像分割的一个小demo

   # 一、项目背景,
    "医学影像检测任务重，复杂，如果能用计算机辅助检测，结果必定事半功倍",
   
   # 二、数据集简介,
    
    "江苏省大数据开发与应用医疗卫生赛道数据集",
   
   ## 1.数据加载和预处理,
    
    "构建训练集",
    train_transforms = [,
       T.RandomHorizontalFlip(),   水平翻转",
        T.RandomVerticalFlip(),  垂直翻转",
        T.RandomRotation(),   随机旋转",
        T.RandomScaleAspect(),   随机缩放",
       T.RandomDistort(),   随机扭曲",
       T.Resize(target_size=(256, 256))   这里为了加快速度，改为256x256\n",
       T.Normalize()   归一化",
   ],
    train_dataset = Dataset(
        transforms=train_transforms,
     dataset_root='train',
        num_classes=2,
        mode='train',
        train_path='train/train_list.txt',
       #separator=' ',
    )
    "构建验证集",
    val_transforms = [,
       T.Resize(target_size=(256, 256)),
        T.Normalize()
    ],
    val_dataset = Dataset(
        transforms=val_transforms,
        dataset_root='train',
        num_classes=2,
        mode='val',
        val_path='train/val_list.txt',
        separator=' ',
    )
    
    
   # 三、模型选择和开发
    
    Unet3+网络：
    train(
       model=unet_3p_model,
        train_dataset=train_dataset,
       val_dataset=val_dataset,
        optimizer=u3p_optimizer,
        save_dir='output_u3p',
        iters=iters,
        batch_size=12,
        save_interval=int(iters/5),
        log_iters=100,
        num_workers=0,
        losses=losses,
       use_vdl=True)
    
   mobilenetv3 分类模型：
    
    num_classes = len(train_dataset.labels)
    model = pdx.cls.MobileNetV3_large_ssld(num_classes=num_classes)
    model.train(num_epochs=12,
                train_dataset=train_dataset,
                train_batch_size=32,
                eval_dataset=eval_dataset,
               lr_decay_epochs=[6, 8],
                save_interval_epochs=1,
                learning_rate=0.00625,
               save_dir='output/mobilenetv3_large_ssld',
               use_vdl=True)
    
   # 四、效果展示,
    
    ![image](https://user-images.githubusercontent.com/74948632/212295103-ee0a36ac-09b9-4588-bb31-2aca2d4804ac.png)
,
    
   ![image](https://user-images.githubusercontent.com/74948632/212295151-4b0aa80a-7829-4e5c-9ba2-ab7c2ecda63e.png)
,
    
    
   # 五、总结与升华,
    
    "医学图像分割初步尝试，继续加油\n",
    
   # 个人简介,
    
    "西南大学2019级本科生，医疗图像方向，持续努力"
   ]
