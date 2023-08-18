import matplotlib.pyplot as plt
from utils.output_manager import OutputManager
import statistics as stats

def transpose_2dlist(mat):
    assert sum([len(l) for l in mat]) == len(mat[0])*len(mat)
    ret = []
    for i in range(len(mat[0])):
        ret.append([])
        for j in range(len(mat)):
            ret[i].append(mat[j][i])
    return ret

def calc_means_stdevs(accs_list):
    tr = transpose_2dlist(accs_list)
    means = [stats.mean(l) for l in tr]
    stdevs = [stats.stdev(l) for l in tr]
    return means, stdevs

def load_data(output_dir, json_name, exp_name, prefix_list,
              label='undef', color='undef', linestyle='-'):
    accs_list = []
    for prefix in prefix_list:
        outman = OutputManager(output_dir, exp_name, prefix_hashing=False)
        results = outman.load_json(json_name, prefix=prefix)
        accs = results['val_accs']
        accs_list.append(accs)
    means, stdevs = calc_means_stdevs(accs_list)

    default_marker = 'o'
    default_markersize = 3.5
    default_linewidth = 1.5
    data = {
        'label': label,
        'means': means,
        'stdevs': stdevs,
        'color': color,
        'alpha': 0.1,
        'marker': default_marker,
        'markersize': default_markersize,
        'linewidth': default_linewidth,
        'linestyle': linestyle,
    }
    return data

def section_4_2_imagenet_cub_resnet18(cfg, outman, prefix, gpu_id):
    datas = []

    # ==== Load data ====
    exp_name = 'transfer_imagenet-cub_resnet18_usetarget'
    label = 'Oracle Transfer'
    color = 'C3'
    linestyle = '-.'
    prefix_list = ['0e9075eca7f28c55fdc0505cfde0f6fe',
                    'f1f5787e9d2e63f1f58f0014cddca7e9',
                    'a39ea976c300060693a432dbf468056a',]
    data = load_data(cfg['output_dir'], 'ft_results', exp_name, prefix_list, color=color, label=label, linestyle=linestyle)
    datas.append(data)

    exp_name = 'transfer_imagenet-cub_resnet18_noperm'
    label = 'Naive Transfer'
    color = 'C1'
    linestyle = '-'
    prefix_list = ['0e9075eca7f28c55fdc0505cfde0f6fe',
                    'f1f5787e9d2e63f1f58f0014cddca7e9',
                    'a39ea976c300060693a432dbf468056a',]
    data = load_data(cfg['output_dir'], 'ft_results', exp_name, prefix_list, color=color, label=label, linestyle=linestyle)
    datas.append(data)

    exp_name = 'transfer_imagenet-cub_resnet18'
    label = 'GMT Transfer'
    color = 'C0'
    linestyle = '-'
    prefix_list = ['0e9075eca7f28c55fdc0505cfde0f6fe',
                    'f1f5787e9d2e63f1f58f0014cddca7e9',
                    'a39ea976c300060693a432dbf468056a',]
    data = load_data(cfg['output_dir'], 'ft_results', exp_name, prefix_list, color=color, label=label, linestyle=linestyle)
    datas.append(data)

    exp_name = 'transfer_imagenet-cub_resnet18_fast'
    label = 'FGMT Transfer'
    color = 'C2'
    linestyle = '-'
    prefix_list = ['0e9075eca7f28c55fdc0505cfde0f6fe',
                    'f1f5787e9d2e63f1f58f0014cddca7e9',
                    'a39ea976c300060693a432dbf468056a',]
    data = load_data(cfg['output_dir'], 'ft_results', exp_name, prefix_list, color=color, label=label, linestyle=linestyle)
    datas.append(data)
    # ====================

    # ===== Meta Data =====
    xs = list(range(1, len(datas[0]['means'])+1))
    name = f'section_4-2_imagenet-cub_resnet18'
    #title = f'Random Init → CIFAR-10 (resnet18)'
    #train_epoch_15 = # TODO
    #train_epoch_17 = # TODO
    # ================

    for d in datas:
        assert len(datas[0]['means']) == len(d['means'])

    filepath = outman.get_abspath(prefix='plot.finetuning', ext='pdf', name=name)
    fig = plt.figure(figsize=(4.8,4.8))
    ax = fig.add_subplot(1,1,1)
    #ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy')
    ax.set_xticks(xs)

    x_min = xs[0]-0.5
    x_max = xs[-1]+0.5
    ax.set_xlim(x_min, x_max)
    ax.grid(color='lightgray')

    for data in datas:
        ys = data['means']
        devs = data['stdevs']
        color = data['color']
        label = data['label']
        alpha = data['alpha']
        marker = data['marker']
        markersize = data['markersize']
        linewidth = data['linewidth']
        linestyle = data['linestyle']
        ax.plot(xs, ys,
                label=label, linewidth=linewidth, linestyle=linestyle,
                marker=marker, markersize=markersize,
                color=color)
        ax.fill_between(xs, [y - s for y, s in zip(ys, devs)], [y + s for y, s in zip(ys, devs)], alpha=alpha, color=color)

    #ax.plot([x_min, x_max], [train_epoch_17, train_epoch_17], "black", linestyle='dashed', label='Source (epoch=17)')
    #ax.plot([x_min, x_max], [train_epoch_15, train_epoch_15], "black", linestyle='dotted', label='Source (epoch=15)')

    ax.legend()

    fig.savefig(filepath, format="pdf", bbox_inches='tight')
    print(filepath)
