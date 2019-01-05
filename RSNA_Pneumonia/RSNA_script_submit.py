# -*- coding: utf-8 -*-
"""
Created on 10/24/18
@author: MRChou

Scenario: 
"""

import pandas
import pydicom as dcm

probability = '/archive/RSNA/submissions/PNASnet_output.csv'
probability = pandas.read_csv(probability, header=None, index_col=0)
probability = probability.to_dict()[1]


# bbox notation: {prob} {xmin} {ymin} {width} {height}
unet = '/archive/RSNA/submissions/submit_unet_0.7.csv'
unet = pandas.read_csv(unet, header=None, index_col=0)
unet = unet.to_dict()[1]

image_id = '/rawdata/RSNA_Pneumonia/stage_2_sample_submission.csv'
image_id = pandas.read_csv(image_id)
image_id = image_id['patientId']

if __name__ == '__main__':

    def validbox(box):
        return (0 < box[1] + box[3] < 1024) and (0 < box[2] + box[4] < 1024)

    def is_ap_view(img_id):
        img_path = '/rawdata/RSNA_Pneumonia/imgs_test/'
        ds = dcm.dcmread(img_path + img_id + '.dcm')
        return ds.ViewPosition == 'AP'


    prob_thres = 0.35
    with open('/archive/RSNA/submissions/stage2_submit_035-07.csv', 'w') as fout:
        fout.write('patientId,PredictionString\n')
        for img_id in image_id:
            fout.write(img_id+', ')

            l_prob = probability[img_id + '_left']
            r_prob = probability[img_id + '_right']

            if is_ap_view(img_id):
                if l_prob > prob_thres:
                    l_box = [int(i) for i in unet[img_id + '_left'].split()]
                    if validbox(l_box):
                        fout.write(
                            '{prob} {xmin} {ymin} {width} {height} '.format(
                                prob = l_box[0],
                                xmin=l_box[1],
                                ymin=l_box[2],
                                width=l_box[3],
                                height=l_box[4]
                            ))

                if r_prob > prob_thres:
                    r_box = [int(i) for i in unet[img_id + '_right'].split()]
                    r_box[1] += 512
                    if validbox(r_box):
                        fout.write(
                            '{prob} {xmin} {ymin} {width} {height} '.format(
                                prob = r_box[0],
                                xmin=r_box[1],
                                ymin=r_box[2],
                                width=r_box[3],
                                height=r_box[4]
                            ))

            fout.write('\n')
