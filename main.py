
import os

from torch.optim.lr_scheduler import ConstantLR, SequentialLR
from dependency import _C as cfg
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import random
import shutil
from sched import scheduler
import click
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import *
from model import *
from dependency import *
from torch import optim
from dataloader import generate_dataloader
import torch
import numpy as np
import swanlab
import warnings
from utils import count_parameters_flops
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from test import predict

def train(net, train_dataloader, scaler):
    torch.autograd.set_detect_anomaly(True)
    net.set_mode('train')
    train_loss = 0
    train_dia_acc = 0
    train_sps_acc = 0
    # train_mel_acc = 0
    weight_derm = cfg.WEIGHT_LIST[0]
    weight_clinic = cfg.WEIGHT_LIST[1]
    weight_fusion = cfg.WEIGHT_LIST[2]
    weight_head = cfg.WEIGHT_LIST[3]

    for index, (clinic_image, derm_image, label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        clinic_image = clinic_image.cuda()
        derm_image = derm_image.cuda()
        # meta_data = meta_data.cuda()

        # Diagostic label
        diagnosis_label = label[0].long().cuda()
        # Seven-Point Checklikst labels
        pn_label = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label = label[7].long().cuda()

        melanoma_label = [1 if d == 2 else 0 for d in diagnosis_label]
        melanoma_label = torch.tensor(melanoma_label).cuda()
        label = [diagnosis_label, pn_label, str_label, pig_label, rs_label, dag_label, bwv_label, vs_label]

        with autocast():
            [
                ((logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm,
                  logit_bwv_derm, logit_vs_derm),
                 (logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic,
                  logit_bwv_clic, logit_vs_clic),
                 (logit_diagnosis_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion,
                  logit_dag_fusion,
                  logit_bwv_fusion, logit_vs_fusion),
                 (logit_head, logit_pn_head, logit_str_head, logit_pig_head, logit_rs_head, logit_dag_head,
                  logit_bwv_head,
                  logit_vs_head),
                 ),
                # align_loss
                # (kg_0, kg_1, kg_2, kg_3)
                # kg_logit,
             # (embeddings_derm, embeddings_clinic, embeddings_fusion, embeddings_head)
            ] = net((clinic_image, derm_image))


            loss_head = torch.true_divide(
                net.criterion_diag(logit_head, diagnosis_label)
                + net.criterion(logit_pn_head, pn_label)
                + net.criterion(logit_str_head, str_label)
                + net.criterion(logit_pig_head, pig_label)
                + net.criterion(logit_rs_head, rs_label)
                + net.criterion(logit_dag_head, dag_label)
                + net.criterion(logit_bwv_head, bwv_label)
                + net.criterion(logit_vs_head, vs_label), 8)
            loss_fusion = torch.true_divide(
                net.criterion_diag(logit_diagnosis_fusion, diagnosis_label)
                + net.criterion(logit_pn_fusion, pn_label)
                + net.criterion(logit_str_fusion, str_label)
                + net.criterion(logit_pig_fusion, pig_label)
                + net.criterion(logit_rs_fusion, rs_label)
                + net.criterion(logit_dag_fusion, dag_label)
                + net.criterion(logit_bwv_fusion, bwv_label)
                + net.criterion(logit_vs_fusion, vs_label), 8)
            loss_clic = torch.true_divide(
                net.criterion_diag(logit_diagnosis_clic, diagnosis_label)
                + net.criterion(logit_pn_clic, pn_label)
                + net.criterion(logit_str_clic, str_label)
                + net.criterion(logit_pig_clic, pig_label)
                + net.criterion(logit_rs_clic, rs_label)
                + net.criterion(logit_dag_clic, dag_label)
                + net.criterion(logit_bwv_clic, bwv_label)
                + net.criterion(logit_vs_clic, vs_label), 8)
            loss_derm = torch.true_divide(
                net.criterion_diag(logit_diagnosis_derm, diagnosis_label)
                + net.criterion(logit_pn_derm, pn_label)
                + net.criterion(logit_str_derm, str_label)
                + net.criterion(logit_pig_derm, pig_label)
                + net.criterion(logit_rs_derm, rs_label)
                + net.criterion(logit_dag_derm, dag_label)
                + net.criterion(logit_bwv_derm, bwv_label)
                + net.criterion(logit_vs_derm, vs_label), 8)


            loss = ((loss_fusion * cfg.WEIGHT_LIST[2] + loss_derm * cfg.WEIGHT_LIST[0] +
                     loss_clic * cfg.WEIGHT_LIST[1] + loss_head * cfg.WEIGHT_LIST[3])
                    / (cfg.WEIGHT_LIST[0] + cfg.WEIGHT_LIST[1] + cfg.WEIGHT_LIST[2] + cfg.WEIGHT_LIST[3]))


            dia_acc_fusion = torch.true_divide(net.metric(logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
            dia_acc_head = torch.true_divide(net.metric(logit_head, diagnosis_label), clinic_image.size(0))
            dia_acc_clic = torch.true_divide(net.metric(logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
            dia_acc_derm = torch.true_divide(net.metric(logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))

            dia_acc = ((dia_acc_fusion * weight_fusion + dia_acc_clic * weight_clinic
                        + dia_acc_derm * weight_derm + dia_acc_head * weight_head)
                       / ( cfg.WEIGHT_LIST[0] + cfg.WEIGHT_LIST[1] + cfg.WEIGHT_LIST[2] + cfg.WEIGHT_LIST[3]))

            sps_acc_head = torch.true_divide(net.metric(logit_pn_head, pn_label)
                                               + net.metric(logit_str_head, str_label)
                                               + net.metric(logit_pig_head, pig_label)
                                               + net.metric(logit_rs_head, rs_label)
                                               + net.metric(logit_dag_head, dag_label)
                                               + net.metric(logit_bwv_head, bwv_label)
                                               + net.metric(logit_vs_head, vs_label), 7 * clinic_image.size(0))
            sps_acc_fusion = torch.true_divide(net.metric(logit_pn_fusion, pn_label)
                                               + net.metric(logit_str_fusion, str_label)
                                               + net.metric(logit_pig_fusion, pig_label)
                                               + net.metric(logit_rs_fusion, rs_label)
                                               + net.metric(logit_dag_fusion, dag_label)
                                               + net.metric(logit_bwv_fusion, bwv_label)
                                               + net.metric(logit_vs_fusion, vs_label), 7 * clinic_image.size(0))
            sps_acc_clic = torch.true_divide(net.metric(logit_pn_clic, pn_label)
                                             + net.metric(logit_str_clic, str_label)
                                             + net.metric(logit_pig_clic, pig_label)
                                             + net.metric(logit_rs_clic, rs_label)
                                             + net.metric(logit_dag_clic, dag_label)
                                             + net.metric(logit_bwv_clic, bwv_label)
                                             + net.metric(logit_vs_clic, vs_label), 7 * clinic_image.size(0))
            sps_acc_derm = torch.true_divide(net.metric(logit_pn_derm, pn_label)
                                             + net.metric(logit_str_derm, str_label)
                                             + net.metric(logit_pig_derm, pig_label)
                                             + net.metric(logit_rs_derm, rs_label)
                                             + net.metric(logit_dag_derm, dag_label)
                                             + net.metric(logit_bwv_derm, bwv_label)
                                             + net.metric(logit_vs_derm, vs_label), 7 * clinic_image.size(0))

            sps_acc = ((sps_acc_fusion * weight_fusion + sps_acc_clic * weight_clinic
                        + sps_acc_derm * weight_derm + sps_acc_head * weight_head)
                       / (cfg.WEIGHT_LIST[0] + cfg.WEIGHT_LIST[1] + cfg.WEIGHT_LIST[2] + cfg.WEIGHT_LIST[3]))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_dia_acc += dia_acc.item()
        train_sps_acc += sps_acc.item()

    train_loss = train_loss / (index + 1)
    train_dia_acc = train_dia_acc / (index + 1)
    train_sps_acc = train_sps_acc / (index + 1)

    return train_loss, train_dia_acc, train_sps_acc


def validation(net, val_dataloader):
    net.set_mode('valid')
    val_loss = 0
    val_dia_acc = 0
    vaL_sps_acc = 0
    # val_mel_acc = 0
    weight_derm = cfg.WEIGHT_LIST[0]
    weight_clinic = cfg.WEIGHT_LIST[1]
    weight_fusion = cfg.WEIGHT_LIST[2]
    weight_head = cfg.WEIGHT_LIST[3]

    for index, (clinic_image, derm_image, label) in enumerate(val_dataloader):
        clinic_image = clinic_image.cuda()
        derm_image = derm_image.cuda()
        # meta_data = meta_data.cuda()

        diagnosis_label = label[0].long().cuda()
        pn_label = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label = label[7].long().cuda()
        melanoma_label = [1 if d == 2 else 0 for d in diagnosis_label]
        melanoma_label = torch.tensor(melanoma_label).cuda()
        label = [diagnosis_label, pn_label, str_label, pig_label, rs_label, dag_label, bwv_label, vs_label]

        pn_binary = (pn_label == 2).long()
        str_binary = (str_label == 2).long()
        pig_binary = (pig_label == 2).long()
        rs_binary = (rs_label == 1).long()
        dag_binary = (dag_label == 2).long()
        bwv_binary = (bwv_label == 1).long()
        vs_binary = (vs_label == 2).long()
        seven_point_labels = torch.stack([
            pn_binary,
            str_binary,
            pig_binary,
            rs_binary,
            dag_binary,
            bwv_binary,
            vs_binary], dim=1)  # (32, 7)

        with ((((((((((((((((torch.no_grad())))))))))))))))):

            [
                ((logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm,
                  logit_bwv_derm, logit_vs_derm),
                 (logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic,
                  logit_bwv_clic, logit_vs_clic),
                 (logit_diagnosis_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion,
                  logit_bwv_fusion, logit_vs_fusion),
                 (logit_head, logit_pn_head, logit_str_head, logit_pig_head, logit_rs_head, logit_dag_head,
                  logit_bwv_head,
                  logit_vs_head),
                 ),
            ] = net((clinic_image, derm_image))


            loss_head = torch.true_divide(
                net.criterion_diag(logit_head, diagnosis_label)
                + net.criterion(logit_pn_head, pn_label)
                + net.criterion(logit_str_head, str_label)
                + net.criterion(logit_pig_head, pig_label)
                + net.criterion(logit_rs_head, rs_label)
                + net.criterion(logit_dag_head, dag_label)
                + net.criterion(logit_bwv_head, bwv_label)
                + net.criterion(logit_vs_head, vs_label), 8)
            loss_fusion = torch.true_divide(
                net.criterion_diag(logit_diagnosis_fusion, diagnosis_label)
                + net.criterion(logit_pn_fusion, pn_label)
                + net.criterion(logit_str_fusion, str_label)
                + net.criterion(logit_pig_fusion, pig_label)
                + net.criterion(logit_rs_fusion, rs_label)
                + net.criterion(logit_dag_fusion, dag_label)
                + net.criterion(logit_bwv_fusion, bwv_label)
                + net.criterion(logit_vs_fusion, vs_label), 8)

            loss_clic = torch.true_divide(
                net.criterion_diag(logit_diagnosis_clic, diagnosis_label)
                + net.criterion(logit_pn_clic, pn_label)
                + net.criterion(logit_str_clic, str_label)
                + net.criterion(logit_pig_clic, pig_label)
                + net.criterion(logit_rs_clic, rs_label)
                + net.criterion(logit_dag_clic, dag_label)
                + net.criterion(logit_bwv_clic, bwv_label)
                + net.criterion(logit_vs_clic, vs_label), 8)

            loss_derm = torch.true_divide(
                net.criterion_diag(logit_diagnosis_derm, diagnosis_label)
                + net.criterion(logit_pn_derm, pn_label)
                + net.criterion(logit_str_derm, str_label)
                + net.criterion(logit_pig_derm, pig_label)
                + net.criterion(logit_rs_derm, rs_label)
                + net.criterion(logit_dag_derm, dag_label)
                + net.criterion(logit_bwv_derm, bwv_label)
                + net.criterion(logit_vs_derm, vs_label), 8)


            loss = ((loss_fusion * cfg.WEIGHT_LIST[2] + loss_derm * cfg.WEIGHT_LIST[0] +
                     loss_clic * cfg.WEIGHT_LIST[1] + loss_head * cfg.WEIGHT_LIST[3])
                    / (cfg.WEIGHT_LIST[0] + cfg.WEIGHT_LIST[1] + cfg.WEIGHT_LIST[2] + cfg.WEIGHT_LIST[3]))


            dia_acc_fusion = torch.true_divide(net.metric(logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
            dia_acc_head = torch.true_divide(net.metric(logit_head, diagnosis_label),  clinic_image.size(0))
            dia_acc_clic = torch.true_divide(net.metric(logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
            dia_acc_derm = torch.true_divide(net.metric(logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))
            #

            dia_acc = ((dia_acc_fusion * weight_fusion + dia_acc_clic * weight_clinic
                        + dia_acc_derm * weight_derm + dia_acc_head * weight_head)
                       / (cfg.WEIGHT_LIST[0] + cfg.WEIGHT_LIST[1] + cfg.WEIGHT_LIST[2] + cfg.WEIGHT_LIST[3]))

            sps_acc_head = torch.true_divide(net.metric(logit_pn_head, pn_label)
                                             + net.metric(logit_str_head, str_label)
                                             + net.metric(logit_pig_head, pig_label)
                                             + net.metric(logit_rs_head, rs_label)
                                             + net.metric(logit_dag_head, dag_label)
                                             + net.metric(logit_bwv_head, bwv_label)
                                             + net.metric(logit_vs_head, vs_label), 7 * clinic_image.size(0))
            sps_acc_fusion = torch.true_divide(net.metric(logit_pn_fusion, pn_label)
                                               + net.metric(logit_str_fusion, str_label)
                                               + net.metric(logit_pig_fusion, pig_label)
                                               + net.metric(logit_rs_fusion, rs_label)
                                               + net.metric(logit_dag_fusion, dag_label)
                                               + net.metric(logit_bwv_fusion, bwv_label)
                                               + net.metric(logit_vs_fusion, vs_label), 7 * clinic_image.size(0))
            sps_acc_clic = torch.true_divide(net.metric(logit_pn_clic, pn_label)
                                             + net.metric(logit_str_clic, str_label)
                                             + net.metric(logit_pig_clic, pig_label)
                                             + net.metric(logit_rs_clic, rs_label)
                                             + net.metric(logit_dag_clic, dag_label)
                                             + net.metric(logit_bwv_clic, bwv_label)
                                             + net.metric(logit_vs_clic, vs_label), 7 * clinic_image.size(0))
            sps_acc_derm = torch.true_divide(net.metric(logit_pn_derm, pn_label)
                                             + net.metric(logit_str_derm, str_label)
                                             + net.metric(logit_pig_derm, pig_label)
                                             + net.metric(logit_rs_derm, rs_label)
                                             + net.metric(logit_dag_derm, dag_label)
                                             + net.metric(logit_bwv_derm, bwv_label)
                                             + net.metric(logit_vs_derm, vs_label), 7 * clinic_image.size(0))

            sps_acc = ((sps_acc_fusion * weight_fusion + sps_acc_clic * weight_clinic
                        + sps_acc_derm * weight_derm + sps_acc_head * weight_head)
                       / (cfg.WEIGHT_LIST[0] + cfg.WEIGHT_LIST[1] + cfg.WEIGHT_LIST[2] + cfg.WEIGHT_LIST[3]))


        val_loss += loss.item()
        val_dia_acc += dia_acc.item()
        vaL_sps_acc += sps_acc.item()

    val_loss = val_loss / (index + 1)
    val_dia_acc = val_dia_acc / (index + 1)
    vaL_sps_acc = vaL_sps_acc / (index + 1)


    return val_loss, val_dia_acc, vaL_sps_acc


def run_train(model_name):
    log.info('** start training here! **\n')
    best_mean_acc = 0
    scaler = GradScaler()
    history = {
        'epoch': [],
        'suspicion_logits': [],  # 将记录每个task的logits
        'logit_gate': []
    }

    for epoch in range(cfg.epochs):

        # train_mode
        start_time = time.time()
        train_loss, train_dia_acc, train_sps_acc = train(net, train_dataloader, scaler)
        end_time = time.time()
        log.info(
            'Train: epoch: {}, l_rate:{:>9.7f}, Train Loss: {:.4f}, Train Dia Acc: {:.4f}, Train SPS Acc: {:.4f}, Epoch_Time:{:>5.2f}min\n'.
            format(epoch, optimizer.param_groups[0]['lr'], train_loss, train_dia_acc, train_sps_acc,
                   (end_time - start_time) / 60))
        wandb_log = {}
        wandb_log["train_loss"] = train_loss
        wandb_log["train_dia_acc"] = train_dia_acc
        wandb_log["train_sps_acc"] = train_sps_acc

        # validation mode
        val_loss, val_dia_acc, val_sps_acc= validation(net, val_dataloader)

        val_mean_acc = (val_dia_acc * 1 + val_sps_acc * 7) / 8

        log.info(
            'Valid: epoch: {}, Valid Loss: {:.4f}, Valid Dia Acc: {:.4f}, Valid SPS Acc: {:.4f}, Valid mean Acc: {:.4f}\n'.format(
                epoch, val_loss,
                val_dia_acc,
                val_sps_acc,
                val_mean_acc))
        wandb_log["val_loss"] = val_loss
        wandb_log["val_dia_acc"] = val_dia_acc
        wandb_log["val_sps_acc"] = val_sps_acc
        wandb_log["val_mean_acc"] = val_mean_acc

        swanlab.log(wandb_log)


        if val_mean_acc > best_mean_acc:
            es = 0
            best_mean_acc = val_mean_acc
            old_path = os.path.join(model_dir, "best_model.pth")
            if os.path.exists(old_path):
                os.remove(old_path)
            torch.save(net.state_dict(), os.path.join(model_dir, "best_model.pth"))
            log.info('------------------Best_Epoch:{:>3d},  Best Mean Acc is {:.5f}------------------'.format(epoch,
                                                                                                              best_mean_acc))

        scheduler.step()

    import numpy as np
    np.save(os.path.join(cfg.OUTPUT_DIR, cfg.NAME, 'training_history.npy'), history)
    torch.save(net.state_dict(), os.path.join(model_dir, 'resnet50_model.pth'))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_input_only_loader(original_loader):
    for batch in original_loader:
        inputs = (batch[0], batch[1])  # 只返回前两个元素
        yield inputs


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 忽略冲突
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(3407)
    if cfg.deterministic:
        if cfg.data_mode == 'Normal':
            random_seeds = 170
        elif cfg.data_mode == 'self_evaluated':
            random_seeds = 183

    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes")
    # create logger
    log, out_dir = CraateLogger(cfg.OUTPUT_DIR, cfg.NAME)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        log.info(
            "This directory has already existed, Please remember to modify your cfg.NAME"
        )
        if not click.confirm(
                "\033[1;31;40mContinue and override the former directory?\033[0m",
                default=False,
        ):
            exit(0)
        shutil.rmtree(code_dir)

    print("=> output model will be saved in {}".format(model_dir))
    this_dir = os.path.dirname(__file__)
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", ".pyth", "*build*", "*output*", "*datasets*",
        "pretrained_models"
    )
    shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)


    train_dataloader, val_dataloader, test_dataloader, train_num, val_num, test_num = generate_dataloader(cfg.shape,
                                                                                                          cfg.batch_size,
                                                                                                          cfg.num_workers,
                                                                                                          cfg.data_mode)
    print("train num:{}, val num:{}, test num:{}".format(train_num, val_num, test_num))
    timestamp = time.strftime('%m%d%H%M')
    wandb_name = timestamp + cfg.NAME
    tags = []
    tags.append(cfg.NAME)
    swanlab.init(project='7pt', name=wandb_name, tags=tags)


    device = torch.device("cuda")
    net = MLCNN(class_list, device=device).cuda()
    optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    flops, params = count_parameters_flops(copy.deepcopy(net))
    print(f"运算量：{flops}, 参数量：{params}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01, last_epoch=-1,
                                                           verbose=False)
    run_train(cfg.NAME)



