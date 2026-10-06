import csv
import os
import time
import warnings

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score,recall_score,precision_score, confusion_matrix,classification_report

from tqdm import tqdm

from dependency import _C as cfg
from dataloader import generate_dataloader
from utils import encode_meta_label, encode_test_label, Logger
from evaluate import FusionMatrix, get_roc_auc
from model import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from plotting_utils import *

normlazing = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])
def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def predict(net, test_index_list, df, weight_list, TTA=4, size=224, device=torch.device("cpu")):
    test_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "test")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    # log = Logger()
    # log.open(os.path.join(test_dir , 'log.test_result.txt'), mode='w')
    # log.write('\n--- [START %s] %s\n\n' % ('IDENTIFIER', '-' * 64))
    metric_csv_fp = open(os.path.join(test_dir, 'test_result.csv'), 'w')
    metric_csv_writer = csv.writer(metric_csv_fp)

    net.set_mode('valid')

    # --- 1. 创建用于保存 .npy 文件的目录 ---
    results_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "test_predictions")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Prediction results will be saved in: {results_dir}")

    chain_mel_correct = 0

    # 7-point score
    # prob
    # 1 pigment_network
    pn_prob_typ_list = [];
    pn_prob_asp_list = [];
    pn_pred_list = [];
    pn_prob_asb_list = [];
    pn_prob_list = []
    # 2 streak
    str_prob_asb_list = [];
    str_prob_reg_list = [];
    str_prob_irg_list = [];
    str_pred_list = [];
    str_prob_list = []

    # 3 pigmentation
    pig_prob_asb_list = [];
    pig_prob_reg_list = [];
    pig_prob_irg_list = [];
    pig_pred_list = []
    pig_prob_list = []

    # 4 regression structure
    rs_prob_asb_list = [];
    rs_prob_prs_list = [];
    rs_pred_list = []
    rs_prob_list = []

    # 5 dots and globules
    dag_prob_asb_list = [];
    dag_prob_reg_list = [];
    dag_prob_irg_list = [];
    dag_pred_list = []
    dag_prob_list = []

    # 6 blue whitish veil l
    bwv_prob_asb_list = [];
    bwv_prob_prs_list = [];
    bwv_pred_list = []
    bwv_prob_list = []

    # 7vascular structure
    vs_prob_asb_list = [];
    vs_prob_reg_list = [];
    vs_prob_irg_list = [];
    vs_pred_list = []
    vs_prob_list = []

    # label
    # 1 pigment_network
    pn_label_typ_list = [];
    pn_label_asp_list = [];
    pn_label_list = [];
    pn_label_asb_list = []
    # 2 streak
    str_label_asb_list = [];
    str_label_reg_list = [];
    str_label_irg_list = [];
    str_label_list = []
    # 3 pigmentation
    pig_label_asb_list = [];
    pig_label_reg_list = [];
    pig_label_irg_list = [];
    pig_label_list = []
    # 4 regression structure
    rs_label_asb_list = [];
    rs_label_prs_list = [];
    rs_label_list = []
    # 5 dots and globules
    dag_label_asb_list = [];
    dag_label_reg_list = [];
    dag_label_irg_list = [];
    dag_label_list = []
    # 6 blue whitish veil l
    bwv_label_asb_list = [];
    bwv_label_prs_list = [];
    bwv_label_list = []
    # 7vascular structure
    vs_label_asb_list = [];
    vs_label_reg_list = [];
    vs_label_irg_list = [];
    vs_label_list = []

    # total
    pred_list = [];
    prob_list = [];
    gt_list = []

    # diagnositi_prob and diagnositic_label
    nevu_prob_list = [];
    bcc_prob_list = [];
    mel_prob_list = [];
    misc_prob_list = [];
    sk_prob_list = []
    nevu_label_list = [];
    bcc_label_list = [];
    mel_label_list = [];
    misc_label_list = [];
    sk_label_list = []
    seven_point_feature_list = []

    # 添加标准差计算
    timings = []
    for index_num in tqdm(test_index_list):
        img_info = df[index_num:index_num + 1]
        clinic_path = img_info['clinic']
        dermoscopy_path = img_info['derm']
        # source_dir = '../release_v0/release_v0/images/'
        clinic_img = cv2.imread(source_dir + clinic_path[index_num])
        dermoscopy_img = cv2.imread(source_dir + dermoscopy_path[index_num])

        meta_data = encode_meta_label(img_info, index_num)
        # encode label
        [diagnosis_label, pigment_network_label, streaks_label, pigmentation_label, regression_structures_label,
         dots_and_globules_label, blue_whitish_veil_label, vascular_structures_label], [diagnosis_label_one_hot,
                                                                                        pigment_network_label_one_hot,
                                                                                        streaks_label_one_hot,
                                                                                        pigmentation_label_one_hot,
                                                                                        regression_structures_label_one_hot,
                                                                                        dots_and_globules_label_one_hot,
                                                                                        blue_whitish_veil_label_one_hot,
                                                                                        vascular_structures_label_one_hot] \
            = encode_test_label(img_info, index_num)

        if TTA == 0:
            meta_data = torch.from_numpy(np.array([meta_data]))
        elif TTA == 4:
            meta_data = torch.from_numpy(np.array([meta_data, meta_data, meta_data, meta_data]))
        elif TTA == 6:
            meta_data = torch.from_numpy(np.array([meta_data, meta_data, meta_data, meta_data, meta_data, meta_data]))

        clinic_img = cv2.resize(clinic_img, (size, size))
        clinic_img_hf = cv2.flip(clinic_img, 0)
        clinic_img_vf = cv2.flip(clinic_img, 1)
        clinic_img_vhf = cv2.flip(clinic_img, -1)
        clinic_img_90 = cv2.rotate(clinic_img, 0)
        clinic_img_270 = cv2.rotate(clinic_img, 2)

        dermoscopy_img = cv2.resize(dermoscopy_img, (size, size))
        dermoscopy_img_hf = cv2.flip(dermoscopy_img, 0)
        dermoscopy_img_vf = cv2.flip(dermoscopy_img, 1)
        dermoscopy_img_vhf = cv2.flip(dermoscopy_img, -1)
        dermoscopy_img_90 = cv2.rotate(dermoscopy_img, 0)
        dermoscopy_img_270 = cv2.rotate(dermoscopy_img, 2)

        clinic_img_total = np.array([clinic_img])
        dermoscopy_img_total = np.array([dermoscopy_img])
        if TTA == 4:
            dermoscopy_img_total = np.array([dermoscopy_img, dermoscopy_img_hf, dermoscopy_img_vf, dermoscopy_img_vhf])
            clinic_img_total = np.array([clinic_img, clinic_img_hf, clinic_img_vf, clinic_img_vhf])
        elif TTA == 6:
            dermoscopy_img_total = np.array([dermoscopy_img, dermoscopy_img_hf,
                                             dermoscopy_img_vf, dermoscopy_img_vhf, dermoscopy_img_90,
                                             dermoscopy_img_270])
            clinic_img_total = np.array(
                [clinic_img, clinic_img_hf, clinic_img_vf, clinic_img_vhf, clinic_img_90, clinic_img_270])

        dermoscopy_img_tensor = torch.from_numpy(
            np.transpose(dermoscopy_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255
        clinic_img_tensor = torch.from_numpy(np.transpose(clinic_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255

        # dermoscopy_img_tensor = normlazing(image=dermoscopy_img)["image"]
        # clinic_img_tensor = normlazing(image=clinic_img)["image"]
        # dermoscopy_img_tensor = dermoscopy_img_tensor.unsqueeze(0)
        # clinic_img_tensor = clinic_img_tensor.unsqueeze(0)

        start = time.perf_counter()
        [
        #(
        # (logit_diagnosis11, logit_pn11, logit_str11, logit_pig11, logit_rs11, logit_dag11, logit_bwv11, logit_vs11),  # derm
        # (logit_diagnosis22, logit_pn22, logit_str22, logit_pig22, logit_rs22, logit_dag22, logit_bwv22, logit_vs22),  # clini
        (logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv, logit_vs),  # fusion
         # (logit_head, logit_pn_head, logit_str_head, logit_pig_head, logit_rs_head, logit_dag_head, logit_bwv_head, logit_vs_head)
        # ),
        # (kg_0, kg_1, kg_2, kg_3)
        # ortho_loss1,
            # ortho_loss2,
            # contrastive_loss,
            # l1_loss1,
            # l1_loss2,
            # kg_loss

        ] = net(
            ((clinic_img_tensor).cuda(), dermoscopy_img_tensor.cuda())
            # dermoscopy_img_tensor.cuda()
        )


        # CCNet
        # [(logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag,
        #   logit_bwv, logit_vs),
        #  ortho_loss1, contrastive_loss, l1_loss,
        #  chain_out] = net(
        #     (clinic_img_tensor.cuda(), dermoscopy_img_tensor.cuda()), true_major_labels=None)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000)

        # # 可视化特征解耦
        # plt.scatter(shared_feat[:, 0], shared_feat[:, 1], c='b', label='Shared')
        # plt.scatter(spec_feat[:, 0], spec_feat[:, 1], c='r', label='Specific')
        # plt.title("Orthogonal Feature Space")
        # plt.legend()



        weight = weight_list[2]  # fusion
        weight_1 = weight_list[1]  # clini
        weight_2 = weight_list[0]  # derm
        weight_3 = weight_list[3]  # head

        # 4输出
        # logit_diagnosis = (weight_1 * logit_diagnosis11 + weight * logit_diagnosis + weight_2 * logit_diagnosis22 + weight_3 * logit_head)/ ( weight_1 + weight_2 + weight + weight_3)
        # logit_pn = (weight_1 * logit_pn11 + weight * logit_pn + weight_2 * logit_pn22 + weight_3 * logit_pn_head)/ (weight_1 + weight_2 + weight + weight_3)
        # logit_str = (weight_1 * logit_str11 + weight * logit_str + weight_2 * logit_str22 + weight_3 * logit_str_head)/ ( weight_1 + weight_2 + weight + weight_3)
        # logit_pig = (weight_1 * logit_pig11 + weight * logit_pig + weight_2 * logit_pig22 + weight_3 * logit_pig_head)/ ( weight_1 + weight_2 + weight + weight_3)
        # logit_rs = (weight_1 * logit_rs11 + weight * logit_rs + weight_2 * logit_rs22 + weight_3 * logit_rs_head)/ ( weight_1 + weight_2 + weight + weight_3)
        # logit_dag = (weight_1 * logit_dag11 + weight * logit_dag + weight_2 * logit_dag22 + weight_3 * logit_dag_head)/ ( weight_1 + weight_2 + weight + weight_3)
        # logit_bwv = (weight_1 * logit_bwv11 + weight * logit_bwv + weight_2 * logit_bwv22 + weight_3 * logit_bwv_head)/ ( weight_1 + weight_2 + weight + weight_3)
        # logit_vs = (weight_1 * logit_vs11 + weight * logit_vs + weight_2 * logit_vs22 + weight_3 * logit_vs_head)/ ( weight_1 + weight_2 + weight + weight_3)

        # 3输出
        # logit_diagnosis =  (logit_diagnosis11 * weight_2 + logit_diagnosis * weight + logit_diagnosis22 * weight_1) / ( weight_1 + weight_2 + weight)
        # logit_pn = (logit_pn11* weight_2 + logit_pn * weight + logit_pn22 * weight_1) / ( weight_1 + weight_2 + weight)
        # logit_str = (logit_str11* weight_2 + logit_str * weight + logit_str22 * weight_1) / ( weight_1 + weight_2 + weight)
        # logit_pig = (logit_pig11* weight_2 + logit_pig * weight + logit_pig22 * weight_1) / ( weight_1 + weight_2 + weight)
        # logit_rs = (logit_rs11* weight_2 + logit_rs * weight + logit_rs22 * weight_1) / ( weight_1 + weight_2 + weight)
        # logit_dag = (logit_dag11* weight_2 + logit_dag * weight + logit_dag22 * weight_1) / ( weight_1 + weight_2 + weight)
        # logit_bwv = (logit_bwv11* weight_2 + logit_bwv * weight + logit_bwv22 * weight_1) / ( weight_1 + weight_2 + weight)
        # logit_vs = (logit_vs11* weight_2 + logit_vs * weight + logit_vs22 * weight_1) / ( weight_1 + weight_2 + weight)

        # 2输出
        # logit_diagnosis = logit_diagnosis11 * weight_2 + logit_diagnosis22 * weight_1
        # logit_pn = logit_pn11 * weight_2 + logit_pn22 * weight_1
        # logit_str = logit_str11 * weight_2 + logit_str22 * weight_1
        # logit_pig = logit_pig11 * weight_2 + logit_pig22 * weight_1
        # logit_rs = logit_rs11 * weight_2 + logit_rs22 * weight_1
        # logit_dag = logit_dag11 * weight_2 + logit_dag22 * weight_1
        # logit_bwv = logit_bwv11 * weight_2 + logit_bwv22 * weight_1
        # logit_vs = logit_vs11 * weight_2 + logit_vs22 * weight_1

        # logit_diagnosis = (logit_diagnosis11 + logit_diagnosis) / 2
        # logit_pn = (logit_pn11 + logit_pn) / 2
        # logit_str = (logit_str11 + logit_str ) / 2
        # logit_pig = (logit_pig11 + logit_pig ) / 2
        # logit_rs = (logit_rs11 + logit_rs) / 2
        # logit_dag = (logit_dag11 + logit_dag)/2
        # logit_bwv = (logit_bwv11 + logit_bwv )/ 2
        # logit_vs =  (logit_vs11 + logit_vs)/2



        # diagnositic_pred
        # major_mask = (chain_out[:, [0, 1, 2]] > 0.5).float()  # 生成二进制掩码
        # major_scores = major_mask.sum(dim=1) * 2  # 对有效项求和并乘以权重
        # minor_mask = (chain_out[:, [3, 4, 5, 6]] > 0.5).float()  # 生成二进制掩码
        # minor_scores = minor_mask.sum(dim=1)
        # total_scores = major_scores + minor_scores
        # diag_preds = (total_scores >= cfg.threshold).int() # 0或1
        #
        # chain_mel_correct += (torch.tensor(diag_preds) == diagnosis_label_one_hot[2]).float()
        # chain_mel_prob = torch.sigmoid(total_scores - cfg.threshold) # （0，1）
        #
        # logit_diagnosis[:, 2] = (chain_mel_prob + logit_diagnosis[:, 2]) / 2.0

        pred = softmax(logit_diagnosis.detach().cpu().numpy());
        pred = np.mean(pred, 0);
        # pred[2] = (pred[2] + chain_mel_prob) / 2.0;
        pred_ = np.argmax(pred)
        nevu_prob = pred[0];
        bcc_prob = pred[1];
        mel_prob = pred[2]
        misc_prob = pred[3];
        sk_prob = pred[4]


        # pn_prob
        pn_pred = softmax(logit_pn.detach().cpu().numpy());
        pn_pred = np.mean(pn_pred, 0);
        pn_pred_ = np.argmax(pn_pred)
        pn_prob_asb = pn_pred[0];
        pn_prob_typ = pn_pred[1];
        pn_prob_asp = pn_pred[2];

        # str_prob
        str_pred = softmax(logit_str.detach().cpu().numpy());
        str_pred = np.mean(str_pred, 0);
        str_pred_ = np.argmax(str_pred)
        str_prob_asb = str_pred[0];
        str_prob_reg = str_pred[1];
        str_prob_irg = str_pred[2]
        # pig_prob
        pig_pred = softmax(logit_pig.detach().cpu().numpy());
        pig_pred = np.mean(pig_pred, 0);
        pig_pred_ = np.argmax(pig_pred)
        pig_prob_asb = pig_pred[0];
        pig_prob_reg = pig_pred[1];
        pig_prob_irg = pig_pred[2]
        # rs_prob
        rs_pred = softmax(logit_rs.detach().cpu().numpy());
        rs_pred = np.mean(rs_pred, 0);
        rs_pred_ = np.argmax(rs_pred)

        rs_prob_asb = rs_pred[0];
        rs_prob_prs = rs_pred[1]
        # dag_prob
        dag_pred = softmax(logit_dag.detach().cpu().numpy());
        dag_pred = np.mean(dag_pred, 0);
        dag_pred_ = np.argmax(dag_pred)
        dag_prob_asb = dag_pred[0];
        dag_prob_reg = dag_pred[1];
        dag_prob_irg = dag_pred[2]
        # bwv_prob
        bwv_pred = softmax(logit_bwv.detach().cpu().numpy());
        bwv_pred = np.mean(bwv_pred, 0);
        bwv_pred_ = np.argmax(bwv_pred)
        bwv_prob_asb = bwv_pred[0];
        bwv_prob_prs = bwv_pred[1]
        # vs_prob
        vs_pred = softmax(logit_vs.detach().cpu().numpy());
        vs_pred = np.mean(vs_pred, 0);
        vs_pred_ = np.argmax(vs_pred)
        vs_prob_asb = vs_pred[0];
        vs_prob_reg = vs_pred[1];
        vs_prob_irg = vs_pred[2]

        seven_point_feature_list.append(np.concatenate([pred, pn_pred, str_pred, pig_pred
                                                           , rs_pred, dag_pred, bwv_pred, vs_pred], 0))


        # diagnositic_label
        pred_list.append(pred_);
        prob_list.append(pred);

        gt_list.append(diagnosis_label)
        nevu_prob_list.append(nevu_prob);
        bcc_prob_list.append(bcc_prob);
        mel_prob_list.append(mel_prob);
        misc_prob_list.append(misc_prob);
        sk_prob_list.append(sk_prob)
        nevu_label_list.append(diagnosis_label_one_hot[0]);
        bcc_label_list.append(diagnosis_label_one_hot[1]);
        mel_label_list.append(diagnosis_label_one_hot[2]);
        misc_label_list.append(diagnosis_label_one_hot[3]);
        sk_label_list.append(diagnosis_label_one_hot[4])

        # pn_label
        pn_pred_list.append(pn_pred_);
        pn_prob_list.append(pn_pred);

        pn_label_list.append(pigment_network_label)
        pn_prob_typ_list.append(pn_prob_typ);
        pn_prob_asp_list.append(pn_prob_asp);
        pn_prob_asb_list.append(pn_prob_asb);

        pn_label_asb_list.append(pigment_network_label_one_hot[0]);
        pn_label_typ_list.append(pigment_network_label_one_hot[1]);
        pn_label_asp_list.append(pigment_network_label_one_hot[2]);

        # str_label
        str_pred_list.append(str_pred_);
        str_prob_list.append(str_pred);

        str_label_list.append(streaks_label)
        str_prob_reg_list.append(str_prob_reg);
        str_prob_irg_list.append(str_prob_irg);
        str_prob_asb_list.append(str_prob_asb)
        str_label_asb_list.append(streaks_label_one_hot[0]);
        str_label_reg_list.append(streaks_label_one_hot[1]);
        str_label_irg_list.append(streaks_label_one_hot[2])

        # pig_label
        pig_pred_list.append(pig_pred_);
        pig_prob_list.append(pig_pred);

        pig_label_list.append(pigmentation_label)
        pig_prob_reg_list.append(pig_prob_reg);
        pig_prob_irg_list.append(pig_prob_irg);
        pig_prob_asb_list.append(pig_prob_asb)
        pig_label_asb_list.append(pigmentation_label_one_hot[0]);
        pig_label_reg_list.append(pigmentation_label_one_hot[1]);
        pig_label_irg_list.append(pigmentation_label_one_hot[2])

        # rs_label
        rs_pred_list.append(rs_pred_);
        rs_prob_list.append(rs_pred);

        rs_label_list.append(regression_structures_label)
        rs_prob_asb_list.append(rs_prob_asb);
        rs_prob_prs_list.append(rs_prob_prs)
        rs_label_asb_list.append(regression_structures_label_one_hot[0]);
        rs_label_prs_list.append(regression_structures_label_one_hot[1])

        # dag_label
        dag_pred_list.append(dag_pred_);
        dag_prob_list.append(dag_pred);

        dag_label_list.append(dots_and_globules_label)
        dag_prob_reg_list.append(dag_prob_reg);
        dag_prob_irg_list.append(dag_prob_irg);
        dag_prob_asb_list.append(dag_prob_asb)
        dag_label_asb_list.append(dots_and_globules_label_one_hot[0]);
        dag_label_reg_list.append(dots_and_globules_label_one_hot[1]);
        dag_label_irg_list.append(dots_and_globules_label_one_hot[2])

        # bwv_label
        bwv_pred_list.append(bwv_pred_);
        bwv_prob_list.append(bwv_pred);

        bwv_label_list.append(blue_whitish_veil_label)
        bwv_prob_asb_list.append(bwv_prob_asb);
        bwv_prob_prs_list.append((bwv_prob_prs))
        bwv_label_asb_list.append(blue_whitish_veil_label_one_hot[0]);
        bwv_label_prs_list.append(blue_whitish_veil_label_one_hot[1])

        # vs_label
        vs_pred_list.append(vs_pred_);
        vs_prob_list.append(vs_pred);

        vs_label_list.append(vascular_structures_label)
        vs_prob_reg_list.append(vs_prob_reg);
        vs_prob_irg_list.append(vs_prob_irg);
        vs_prob_asb_list.append(vs_prob_asb)
        vs_label_asb_list.append(vascular_structures_label_one_hot[0]);
        vs_label_reg_list.append(vascular_structures_label_one_hot[1]);
        vs_label_irg_list.append(vascular_structures_label_one_hot[2])


    avg_time = sum(timings) / len(timings)
    std_time = torch.std(torch.tensor(timings)).item()
    print(f"Average: {avg_time:.2f}±{std_time:.2f}")



    pred = np.array(pred_list).squeeze();
    prob = np.array(prob_list).squeeze();


    gt = np.array(gt_list)
    nevu_prob = np.array(nevu_prob_list);
    bcc_prob = np.array(bcc_prob_list);
    mel_prob = np.array(mel_prob_list);
    misc_prob = np.array(misc_prob_list);
    sk_prob = np.array(sk_prob_list)
    nevu_label = np.array(nevu_label_list);
    bcc_label = np.array(bcc_label_list);
    mel_label = np.array(mel_label_list);
    misc_label = np.array(misc_label_list);
    sk_label = np.array(sk_label_list)

    pn_pred = np.array(pn_pred_list).squeeze();
    pn_prob = np.array(pn_prob_list).squeeze();

    pn_gt = np.array(pn_label_list)
    pn_prob_typ = np.array(pn_prob_typ_list);
    pn_prob_asp = np.array(pn_prob_asp_list);
    pn_prob_asb = np.array(pn_prob_asb_list)

    pn_label_typ = np.array(pn_label_typ_list);
    pn_label_asp = np.array(pn_label_asp_list);
    pn_label_asb = np.array(pn_label_asb_list)

    str_pred = np.array(str_pred_list).squeeze();
    str_prob = np.array(str_prob_list).squeeze();

    str_gt = np.array(str_label_list)
    str_prob_asb = np.array(str_prob_asb_list);
    str_prob_reg = np.array(str_prob_reg_list);
    str_prob_irg = np.array(str_prob_irg_list)
    str_label_asb = np.array(str_label_asb_list);
    str_label_reg = np.array(str_label_reg_list);
    str_label_irg = np.array(str_label_irg_list)

    pig_pred = np.array(pig_pred_list).squeeze();
    pig_prob = np.array(pig_prob_list).squeeze();

    pig_gt = np.array(pig_label_list)
    pig_prob_asb = np.array(pig_prob_asb_list);
    pig_prob_reg = np.array(pig_prob_reg_list);
    pig_prob_irg = np.array(pig_prob_irg_list)
    pig_label_asb = np.array(pig_label_asb_list);
    pig_label_reg = np.array(pig_label_reg_list);
    pig_label_irg = np.array(pig_label_irg_list)

    rs_pred = np.array(rs_pred_list).squeeze();
    rs_prob = np.array(rs_prob_list).squeeze();

    rs_gt = np.array(rs_label_list)
    rs_prob_asb = np.array(rs_prob_asb_list);
    rs_prob_prs = np.array(rs_prob_prs_list)
    rs_label_asb = np.array(rs_label_asb_list);
    rs_label_prs = np.array(rs_label_prs_list)

    dag_pred = np.array(dag_pred_list).squeeze();
    dag_prob = np.array(dag_prob_list).squeeze();

    dag_gt = np.array(dag_label_list)
    dag_prob_asb = np.array(dag_prob_asb_list);
    dag_prob_reg = np.array(dag_prob_reg_list);
    dag_prob_irg = np.array(dag_prob_irg_list)
    dag_label_asb = np.array(dag_label_asb_list);
    dag_label_reg = np.array(dag_label_reg_list);
    dag_label_irg = np.array(dag_label_irg_list)

    bwv_pred = np.array(bwv_pred_list).squeeze();
    bwv_prob = np.array(bwv_prob_list).squeeze();

    bwv_gt = np.array(bwv_label_list)
    bwv_prob_asb = np.array(bwv_prob_asb_list);
    bwv_prob_prs = np.array(bwv_prob_prs_list)
    bwv_label_asb = np.array(bwv_label_asb_list);
    bwv_label_prs = np.array(bwv_label_prs_list)

    vs_pred = np.array(vs_pred_list).squeeze();
    vs_prob = np.array(vs_prob_list).squeeze();

    vs_gt = np.array(vs_label_list)
    vs_prob_asb = np.array(vs_prob_asb_list);
    vs_prob_reg = np.array(vs_prob_reg_list);
    vs_prob_irg = np.array(vs_prob_irg_list)
    vs_label_asb = np.array(vs_label_asb_list);
    vs_label_reg = np.array(vs_label_reg_list);
    vs_label_irg = np.array(vs_label_irg_list)

    vs_acc = np.mean(vs_pred == vs_gt)
    bwv_acc = np.mean(bwv_pred == bwv_gt)
    dag_acc = np.mean(dag_pred == dag_gt)
    rs_acc = np.mean(rs_pred == rs_gt)
    pig_acc = np.mean(pig_pred == pig_gt)
    str_acc = np.mean(str_pred == str_gt)
    pn_acc = np.mean(pn_pred == pn_gt)
    diag_acc = np.mean(pred == gt)

    task_avg_acc = (vs_acc + bwv_acc + dag_acc + rs_acc + pig_acc + str_acc + pn_acc + diag_acc) / 8
    print("avg_acc:", task_avg_acc)
    print("diag_acc:", diag_acc)
    metric_csv_writer.writerow(['metrics', 'VS', 'BWV', 'DAG', 'RS', 'PIG', 'STR', 'PN', 'DIAG', 'avg'])
    row = ['ACC', vs_acc, bwv_acc, dag_acc, rs_acc, pig_acc, str_acc, pn_acc, diag_acc, task_avg_acc]
    metric_csv_writer.writerow(row)

    # log.write('-' * 15 + 'ACC' + '-' * 15 + '\n')
    # log.write('avg_acc : {}\n'.format(avg_acc))
    # log.write('vs_acc : {}\n'.format(np.mean(vs_pred == vs_gt)))
    # log.write('bwv_acc : {}\n'.format(np.mean(bwv_pred == bwv_gt)))
    # log.write('dag_acc : {}\n'.format(np.mean(dag_pred == dag_gt)))
    # log.write('rs_acc : {}\n'.format(np.mean(rs_pred == rs_gt)))
    # log.write('pig_acc : {}\n'.format(np.mean(pig_pred == pig_gt)))
    # log.write('str_acc : {}\n'.format(np.mean(str_pred == str_gt)))
    # log.write('pn_acc : {}\n'.format(np.mean(pn_pred == pn_gt)))
    # log.write('diag_acc : {}\n'.format(np.mean(pred == gt)))
    # -------------Diag----------------
    metric_csv_writer.writerow([])
    metric_csv_writer.writerow(['Diag', 'NEVU', 'BCC', 'MEL', 'MISC', 'SK', 'avg'])
    fusion_matrix = FusionMatrix(num_label, label_list)
    fusion_matrix.update(pred, gt)
    nevu_auc = roc_auc_score((np.array(nevu_label) * 1).flatten(), nevu_prob.flatten())
    bcc_auc = roc_auc_score((np.array(bcc_label) * 1).flatten(), bcc_prob.flatten())
    mel_auc = roc_auc_score((np.array(mel_label) * 1).flatten(), mel_prob.flatten())
    misc_auc = roc_auc_score((np.array(misc_label) * 1).flatten(), misc_prob.flatten())
    sk_auc = roc_auc_score((np.array(sk_label) * 1).flatten(), sk_prob.flatten())
    avg_diag_auc = (nevu_auc + bcc_auc + mel_auc + misc_auc + sk_auc) / 5
    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["precision"] = fusion_matrix.get_pre_per_class()
    metrics["specificity"] = fusion_matrix.get_spec_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = [nevu_auc, bcc_auc, mel_auc, misc_auc, sk_auc, avg_diag_auc]
    metrics["fusion_matrix"] = fusion_matrix.matrix
    # auc_mean = np.mean(metrics["roc_auc"])
    row_auc = ['AUC']
    row_auc.extend(metrics["roc_auc"])
    # row_auc.append(auc_mean)
    metric_csv_writer.writerow(row_auc)
    # AP
    average_pre = np.mean(metrics["precision"])
    row_ap = ['Pre']
    row_ap.extend(metrics["precision"])
    row_ap.append(average_pre)
    metric_csv_writer.writerow(row_ap)
    # ACC
    metrics["acc_class"] = fusion_matrix.get_acc_per_class()
    metrics["acc"] = fusion_matrix.get_accuracy()
    avg_acc = np.mean(metrics["acc_class"])
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    row_acc = ['ACC']
    row_acc.extend(metrics["acc_class"])
    row_acc.append(avg_acc)
    metric_csv_writer.writerow(row_acc)
    # SE
    se_mean = np.mean(metrics["sensitivity"])
    row_se = ['SE']
    row_se.extend(metrics["sensitivity"])
    row_se.append(se_mean)
    metric_csv_writer.writerow(row_se)
    # SP
    avg_sp = np.mean(metrics["specificity"])
    row_sp = ['SP']
    row_sp.extend(metrics["specificity"])
    row_sp.append(avg_sp)
    metric_csv_writer.writerow(row_sp)

    print("avg_diag_auc:", avg_diag_auc)
    # log.write('-' * 15 + 'Diag' + '-' * 15 + '\n')
    # log.write('-' * 8 + 'AUC' + '-' * 8 + '\n')
    # log.write('avg_diag_auc : {}\n'.format(avg_diag_auc))
    # log.write('nevu_auc: {}\n'.format(nevu_auc))
    # log.write('bcc_auc: {}\n'.format(bcc_auc))
    # log.write('mel_auc: {}\n'.format(mel_auc))
    # log.write('misc_auc: {}\n'.format(misc_auc))
    # log.write('sk_auc: {}\n'.format(sk_auc))

    # VS
    metric_csv_writer.writerow([])
    metric_csv_writer.writerow(['VS', 'asb', 'reg', 'irg', 'avg'])
    fusion_matrix = FusionMatrix(num_vascular_structures_label, vascular_structures_label_list)
    fusion_matrix.update(vs_pred, vs_gt)
    vs_asb_auc = roc_auc_score((np.array(vs_label_asb) * 1).flatten(), vs_prob_asb.flatten())
    vs_reg_auc = roc_auc_score((np.array(vs_label_reg) * 1).flatten(), vs_prob_reg.flatten())
    vs_irg_auc = roc_auc_score((np.array(vs_label_irg) * 1).flatten(), vs_prob_irg.flatten())

    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["precision"] = fusion_matrix.get_pre_per_class()
    metrics["specificity"] = fusion_matrix.get_spec_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = [vs_asb_auc, vs_reg_auc, vs_irg_auc]
    metrics["fusion_matrix"] = fusion_matrix.matrix
    # AUC
    auc_mean = np.mean(metrics["roc_auc"])
    row_auc = ['AUC']
    row_auc.extend(metrics["roc_auc"])
    row_auc.append(auc_mean)
    metric_csv_writer.writerow(row_auc)
    # AP
    average_pre = np.mean(metrics["precision"])
    row_ap = ['Pre']
    row_ap.extend(metrics["precision"])
    row_ap.append(average_pre)
    metric_csv_writer.writerow(row_ap)
    # ACC
    metrics["acc_class"] = fusion_matrix.get_acc_per_class()
    metrics["acc"] = fusion_matrix.get_accuracy()
    avg_acc = np.mean(metrics["acc_class"])
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    row_acc = ['ACC']
    row_acc.extend(metrics["acc_class"])
    row_acc.append(avg_acc)
    metric_csv_writer.writerow(row_acc)
    # SE
    se_mean = np.mean(metrics["sensitivity"])
    row_se = ['SE']
    row_se.extend(metrics["sensitivity"])
    row_se.append(se_mean)
    metric_csv_writer.writerow(row_se)
    # SP
    avg_sp = np.mean(metrics["specificity"])
    row_sp = ['SP']
    row_sp.extend(metrics["specificity"])
    row_sp.append(avg_sp)
    metric_csv_writer.writerow(row_sp)


    # log.write('-' * 15 + "\n")
    # log.write('vs_asb_auc: {}\n'.format(vs_asb_auc))
    # log.write('vs_reg_auc: {}\n'.format(vs_reg_auc))
    # log.write('vs_irg_auc: {}\n'.format(vs_irg_auc))

    # bwv
    metric_csv_writer.writerow([])
    metric_csv_writer.writerow(['BWV', 'asb', 'prs', 'avg'])
    fusion_matrix = FusionMatrix(num_blue_whitish_veil_label, blue_whitish_veil_label_list)
    fusion_matrix.update(bwv_pred, bwv_gt)
    bwv_asb_auc = roc_auc_score((np.array(bwv_label_asb) * 1).flatten(), bwv_prob_asb.flatten())
    bwv_prs_auc = roc_auc_score((np.array(bwv_label_prs) * 1).flatten(), bwv_prob_prs.flatten())
    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["precision"] = fusion_matrix.get_pre_per_class()
    metrics["specificity"] = fusion_matrix.get_spec_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = [bwv_asb_auc, bwv_prs_auc]
    metrics["fusion_matrix"] = fusion_matrix.matrix
    # AUC
    auc_mean = np.mean(metrics["roc_auc"])
    row_auc = ['AUC']
    row_auc.extend(metrics["roc_auc"])
    row_auc.append(auc_mean)
    metric_csv_writer.writerow(row_auc)
    # AP
    average_pre = np.mean(metrics["precision"])
    row_ap = ['Pre']
    row_ap.extend(metrics["precision"])
    row_ap.append(average_pre)
    metric_csv_writer.writerow(row_ap)
    # ACC
    metrics["acc_class"] = fusion_matrix.get_acc_per_class()
    metrics["acc"] = fusion_matrix.get_accuracy()
    avg_acc = np.mean(metrics["acc_class"])
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    row_acc = ['ACC']
    row_acc.extend(metrics["acc_class"])
    row_acc.append(avg_acc)
    metric_csv_writer.writerow(row_acc)
    # SE
    se_mean = np.mean(metrics["sensitivity"])
    row_se = ['SE']
    row_se.extend(metrics["sensitivity"])
    row_se.append(se_mean)
    metric_csv_writer.writerow(row_se)
    # SP
    avg_sp = np.mean(metrics["specificity"])
    row_sp = ['SP']
    row_sp.extend(metrics["specificity"])
    row_sp.append(avg_sp)
    metric_csv_writer.writerow(row_sp)

    # log.write('-' * 15 + '\n')
    # log.write('bwv_asb_auc: {}\n'.format(bwv_asb_auc))
    # log.write('bwv_prs_auc: {}\n'.format(bwv_prs_auc))

    # dag
    metric_csv_writer.writerow([])
    metric_csv_writer.writerow(['DAG', 'asb', 'reg', 'irg', 'avg'])
    fusion_matrix = FusionMatrix(num_dots_and_globules_label, dots_and_globules_label_list)
    fusion_matrix.update(dag_pred, dag_gt)
    dag_asb_auc = roc_auc_score((np.array(dag_label_asb) * 1).flatten(), dag_prob_asb.flatten())
    dag_reg_auc = roc_auc_score((np.array(dag_label_reg) * 1).flatten(), dag_prob_reg.flatten())
    dag_irg_auc = roc_auc_score((np.array(dag_label_irg) * 1).flatten(), dag_prob_irg.flatten())
    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["precision"] = fusion_matrix.get_pre_per_class()
    metrics["specificity"] = fusion_matrix.get_spec_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = [dag_asb_auc, dag_reg_auc, dag_irg_auc]
    metrics["fusion_matrix"] = fusion_matrix.matrix
    # AUC
    auc_mean = np.mean(metrics["roc_auc"])
    row_auc = ['AUC']
    row_auc.extend(metrics["roc_auc"])
    row_auc.append(auc_mean)
    metric_csv_writer.writerow(row_auc)
    # AP
    average_pre = np.mean(metrics["precision"])
    row_ap = ['Pre']
    row_ap.extend(metrics["precision"])
    row_ap.append(average_pre)
    metric_csv_writer.writerow(row_ap)
    # ACC
    metrics["acc_class"] = fusion_matrix.get_acc_per_class()
    metrics["acc"] = fusion_matrix.get_accuracy()
    avg_acc = np.mean(metrics["acc_class"])
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    row_acc = ['ACC']
    row_acc.extend(metrics["acc_class"])
    row_acc.append(avg_acc)
    metric_csv_writer.writerow(row_acc)
    # SE
    se_mean = np.mean(metrics["sensitivity"])
    row_se = ['SE']
    row_se.extend(metrics["sensitivity"])
    row_se.append(se_mean)
    metric_csv_writer.writerow(row_se)
    # SP
    avg_sp = np.mean(metrics["specificity"])
    row_sp = ['SP']
    row_sp.extend(metrics["specificity"])
    row_sp.append(avg_sp)
    metric_csv_writer.writerow(row_sp)
    # log.write('-' * 15 + '\n')
    # log.write('dag_asb_auc: {}\n'.format(dag_asb_auc))
    # log.write('dag_reg_auc: {}\n'.format(dag_reg_auc))
    # log.write('dag_irg_auc: {}\n'.format(dag_irg_auc))

    # rs
    metric_csv_writer.writerow([])
    metric_csv_writer.writerow(['RS', 'asb', 'prs', 'avg'])
    fusion_matrix = FusionMatrix(num_regression_structures_label, regression_structures_label_list)
    fusion_matrix.update(rs_pred, rs_gt)
    rs_asb_auc = roc_auc_score((np.array(rs_label_asb) * 1).flatten(), rs_prob_asb.flatten())
    rs_prs_auc = roc_auc_score((np.array(rs_label_prs) * 1).flatten(), rs_prob_prs.flatten())
    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["precision"] = fusion_matrix.get_pre_per_class()
    metrics["specificity"] = fusion_matrix.get_spec_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = [rs_asb_auc, rs_prs_auc]
    metrics["fusion_matrix"] = fusion_matrix.matrix
    # AUC
    auc_mean = np.mean(metrics["roc_auc"])
    row_auc = ['AUC']
    row_auc.extend(metrics["roc_auc"])
    row_auc.append(auc_mean)
    metric_csv_writer.writerow(row_auc)
    # AP
    average_pre = np.mean(metrics["precision"])
    row_ap = ['Pre']
    row_ap.extend(metrics["precision"])
    row_ap.append(average_pre)
    metric_csv_writer.writerow(row_ap)
    # ACC
    metrics["acc_class"] = fusion_matrix.get_acc_per_class()
    metrics["acc"] = fusion_matrix.get_accuracy()
    avg_acc = np.mean(metrics["acc_class"])
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    row_acc = ['ACC']
    row_acc.extend(metrics["acc_class"])
    row_acc.append(avg_acc)
    metric_csv_writer.writerow(row_acc)
    # SE
    se_mean = np.mean(metrics["sensitivity"])
    row_se = ['SE']
    row_se.extend(metrics["sensitivity"])
    row_se.append(se_mean)
    metric_csv_writer.writerow(row_se)
    # SP
    avg_sp = np.mean(metrics["specificity"])
    row_sp = ['SP']
    row_sp.extend(metrics["specificity"])
    row_sp.append(avg_sp)
    metric_csv_writer.writerow(row_sp)
    # log.write('-' * 15 + '\n')
    # log.write('rs_asb_auc: {}\n'.format(rs_asb_auc))
    # log.write('rs_prs_auc: {}\n'.format(rs_prs_auc))

    # pig
    metric_csv_writer.writerow([])
    metric_csv_writer.writerow(['PIG', 'asb', 'reg', 'irg', 'avg'])
    fusion_matrix = FusionMatrix(num_pigmentation_label, pigmentation_label_list)
    fusion_matrix.update(pig_pred, pig_gt)
    pig_asb_auc = roc_auc_score((np.array(pig_label_asb) * 1).flatten(), pig_prob_asb.flatten())
    pig_reg_auc = roc_auc_score((np.array(pig_label_reg) * 1).flatten(), pig_prob_reg.flatten())
    pig_irg_auc = roc_auc_score((np.array(pig_label_irg) * 1).flatten(), pig_prob_irg.flatten())
    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["precision"] = fusion_matrix.get_pre_per_class()
    metrics["specificity"] = fusion_matrix.get_spec_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = [pig_asb_auc, pig_reg_auc, pig_irg_auc]
    metrics["fusion_matrix"] = fusion_matrix.matrix
    # AUC
    auc_mean = np.mean(metrics["roc_auc"])
    row_auc = ['AUC']
    row_auc.extend(metrics["roc_auc"])
    row_auc.append(auc_mean)
    metric_csv_writer.writerow(row_auc)
    # AP
    average_pre = np.mean(metrics["precision"])
    row_ap = ['Pre']
    row_ap.extend(metrics["precision"])
    row_ap.append(average_pre)
    metric_csv_writer.writerow(row_ap)
    # ACC
    metrics["acc_class"] = fusion_matrix.get_acc_per_class()
    metrics["acc"] = fusion_matrix.get_accuracy()
    avg_acc = np.mean(metrics["acc_class"])
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    row_acc = ['ACC']
    row_acc.extend(metrics["acc_class"])
    row_acc.append(avg_acc)
    metric_csv_writer.writerow(row_acc)
    # SE
    se_mean = np.mean(metrics["sensitivity"])
    row_se = ['SE']
    row_se.extend(metrics["sensitivity"])
    row_se.append(se_mean)
    metric_csv_writer.writerow(row_se)
    # SP
    avg_sp = np.mean(metrics["specificity"])
    row_sp = ['SP']
    row_sp.extend(metrics["specificity"])
    row_sp.append(avg_sp)
    metric_csv_writer.writerow(row_sp)
    # str
    metric_csv_writer.writerow([])
    metric_csv_writer.writerow(['STR', 'asb', 'reg', 'irg', 'avg'])
    fusion_matrix = FusionMatrix(num_streaks_label, streaks_label_list)
    fusion_matrix.update(str_pred, str_gt)
    str_asb_auc = roc_auc_score((np.array(str_label_asb) * 1).flatten(), str_prob_asb.flatten())
    str_reg_auc = roc_auc_score((np.array(str_label_reg) * 1).flatten(), str_prob_reg.flatten())
    str_irg_auc = roc_auc_score((np.array(str_label_irg) * 1).flatten(), str_prob_irg.flatten())
    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["precision"] = fusion_matrix.get_pre_per_class()
    metrics["specificity"] = fusion_matrix.get_spec_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = [str_asb_auc, str_reg_auc, str_irg_auc]
    metrics["fusion_matrix"] = fusion_matrix.matrix
    # AUC
    auc_mean = np.mean(metrics["roc_auc"])
    row_auc = ['AUC']
    row_auc.extend(metrics["roc_auc"])
    row_auc.append(auc_mean)
    metric_csv_writer.writerow(row_auc)
    # AP
    average_pre = np.mean(metrics["precision"])
    row_ap = ['Pre']
    row_ap.extend(metrics["precision"])
    row_ap.append(average_pre)
    metric_csv_writer.writerow(row_ap)
    # ACC
    metrics["acc_class"] = fusion_matrix.get_acc_per_class()
    metrics["acc"] = fusion_matrix.get_accuracy()
    avg_acc = np.mean(metrics["acc_class"])
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    row_acc = ['ACC']
    row_acc.extend(metrics["acc_class"])
    row_acc.append(avg_acc)
    metric_csv_writer.writerow(row_acc)
    # SE
    se_mean = np.mean(metrics["sensitivity"])
    row_se = ['SE']
    row_se.extend(metrics["sensitivity"])
    row_se.append(se_mean)
    metric_csv_writer.writerow(row_se)
    # SP
    avg_sp = np.mean(metrics["specificity"])
    row_sp = ['SP']
    row_sp.extend(metrics["specificity"])
    row_sp.append(avg_sp)
    metric_csv_writer.writerow(row_sp)

    # pn
    metric_csv_writer.writerow([])
    metric_csv_writer.writerow(['PN', 'abs', 'typ', 'aty', 'avg'])
    fusion_matrix = FusionMatrix(num_pigment_network_label, pigment_network_label_list)
    fusion_matrix.update(pn_pred, pn_gt)
    pn_typ_auc = roc_auc_score((np.array(pn_label_typ) * 1).flatten(), pn_prob_typ.flatten())
    pn_asp_auc = roc_auc_score((np.array(pn_label_asp) * 1).flatten(), pn_prob_asp.flatten())
    pn_asb_auc = roc_auc_score((np.array(pn_label_asb) * 1).flatten(), pn_prob_asb.flatten())
    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["precision"] = fusion_matrix.get_pre_per_class()
    metrics["specificity"] = fusion_matrix.get_spec_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = [pn_asb_auc, pn_typ_auc, pn_asp_auc]
    metrics["fusion_matrix"] = fusion_matrix.matrix
    # AUC
    auc_mean = np.mean(metrics["roc_auc"])
    row_auc = ['AUC']
    row_auc.extend(metrics["roc_auc"])
    row_auc.append(auc_mean)
    metric_csv_writer.writerow(row_auc)
    # AP
    average_pre = np.mean(metrics["precision"])
    row_ap = ['Pre']
    row_ap.extend(metrics["precision"])
    row_ap.append(average_pre)
    metric_csv_writer.writerow(row_ap)
    # ACC
    metrics["acc_class"] = fusion_matrix.get_acc_per_class()
    metrics["acc"] = fusion_matrix.get_accuracy()
    avg_acc = np.mean(metrics["acc_class"])
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    row_acc = ['ACC']
    row_acc.extend(metrics["acc_class"])
    row_acc.append(avg_acc)
    metric_csv_writer.writerow(row_acc)
    # SE
    se_mean = np.mean(metrics["sensitivity"])
    row_se = ['SE']
    row_se.extend(metrics["sensitivity"])
    row_se.append(se_mean)
    metric_csv_writer.writerow(row_se)
    # SP
    avg_sp = np.mean(metrics["specificity"])
    row_sp = ['SP']
    row_sp.extend(metrics["specificity"])
    row_sp.append(avg_sp)
    metric_csv_writer.writerow(row_sp)


    # log.close()
    # --- 2. 循环结束后，在函数末尾添加保存逻辑 ---
    print("\n--- Saving prediction results to .npy files ---")
    # --- 保存 DIAG 任务的结果 ---
    # a. 将列表转换为Numpy数组
    y_true_diag = np.array(gt_list)
    y_prob_diag = np.array(prob_list)
    y_pred_diag = np.array(pred_list)
    # b. 保存到文件
    np.save(os.path.join(results_dir, "true_labels_diag.npy"), y_true_diag)
    np.save(os.path.join(results_dir, "probabilities_diag.npy"), y_prob_diag)
    np.save(os.path.join(results_dir, "predictions_diag.npy"), y_pred_diag)
    print("DIAG task results saved.")



    return task_avg_acc



if __name__ == "__main__":
    # 忽略 UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    model = MLCNN(class_list, device=device).cuda()
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
    print("have loaded the best model from {}".format(model_path))
    checkpoint = torch.load(
            model_path
        )
    model.load_state_dict(checkpoint,strict=False)
    test_batch_size = cfg.TEST.BATCH_SIZE
    train_dataloader, val_dataloader, test_dataloader, train_num, val_num, test_num = generate_dataloader(cfg.shape,
                                                                                                          test_batch_size,
                                                                                                          cfg.TEST.NUM_WORKERS,
                                                                                                          cfg.data_mode)
    print("train num:{}, val num:{}, test num:{}".format(train_num, val_num, test_num))
    test_index_df = pd.read_csv(test_index_path)
    test_index_list = list(test_index_df['indexes'])
    df = pd.read_csv(img_info_path)

    flops, params = count_parameters_flops(copy.deepcopy(model))
    print(f"运算量：{flops}, 参数量：{params}")
    predict(model, test_index_list, df, weight_list=cfg.WEIGHT_LIST, TTA=4, device=device)