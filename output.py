import numpy as np


def printOuput(output, target, file):

    if len(output.shape) > 1:
        output = output[:, 0]
    if len(target.shape) > 1:
        target = target[:, 0]

    output_flop = output < 500
    output_mild = np.logical_and(500 <= output, output < 1400)
    output_succ = np.logical_and(1400 <= output, output < 5000)
    output_great = np.logical_and(5000 <= output, output < 10000)
    output_viral = output >= 10000

    full_output = np.array([output_flop, output_mild, output_succ, output_great, output_viral])

    target_flop = target < 500
    target_mild = np.logical_and(500 <= target, target < 1400)
    target_succ = np.logical_and(1400 <= target, target < 5000)
    target_great = np.logical_and(5000 <= target, target < 10000)
    target_viral = target >= 10000

    full_target = np.array([target_flop, target_mild, target_succ, target_great, target_viral])

    flop_hit = np.size(np.where(np.logical_and(target < 500, output < 500)))
    mild_success_hit = np.size(np.where(
        np.logical_and(np.logical_and(500 <= target, target < 1400), np.logical_and(500 <= output, output < 1400))))
    success_hit = np.size(np.where(
        np.logical_and(np.logical_and(1400 <= target, target < 5000), np.logical_and(1400 <= output, output < 5000))))
    great_success_hit = np.size(np.where(
        np.logical_and(np.logical_and(5000 <= target, target < 10000), np.logical_and(5000 <= output, output < 10000))))
    viral_hit = np.size(np.where(np.logical_and(target >= 10000, output >= 10000)))

    labels = ['flop', 'mild_success', 'success', 'great_success', 'viral']

    file.write("Flop articles :\n\t"
               "- Nbr in output : {}\n\t"
               "- Nbr in target : {}\n".format(sum(output_flop), sum(target_flop)))
    for i in range(5):
        nbr = sum(np.logical_and(target_flop, full_output[i]))
        file.write("\t- Target flops classified as {} : {}\n".format(labels[i], nbr))


    file.write("\nMild success articles :\n\t"
               "- Nbr in output : {}\n\t"
               "- Nbr in target : {}\n".format(sum(output_mild), sum(target_mild)))
    for i in range(5):
        nbr = sum(np.logical_and(target_mild, full_output[i]))
        file.write("\t- Target mild_successes classified as {} : {}\n".format(labels[i], nbr))


    file.write("\nSuccess articles :\n\t"
               "- Nbr in output : {}\n\t"
               "- Nbr in target : {}\n".format(sum(output_succ), sum(target_succ)))
    for i in range(5):
        nbr = sum(np.logical_and(target_succ, full_output[i]))
        file.write("\t- Target successes classified as {} : {}\n".format(labels[i], nbr))


    file.write("\nGreat success articles :\n\t"
               "- Nbr in output : {}\n\t"
               "- Nbr in target : {}\n".format(sum(output_great), sum(target_great)))
    for i in range(5):
        nbr = sum(np.logical_and(target_great, full_output[i]))
        file.write("\t- Target great_successes classified as {} : {}\n".format(labels[i], nbr))


    file.write("\nViral articles :\n\t"
               "- Nbr in output : {}\n\t"
               "- Nbr in target : {}\n".format(sum(output_viral), sum(target_viral)))
    for i in range(5):
        nbr = sum(np.logical_and(target_viral, full_output[i]))
        file.write("\t- Target virals classified as {} : {}\n".format(labels[i], nbr))
