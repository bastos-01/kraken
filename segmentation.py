import click
import math
import pathlib
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

from PIL import Image

from kraken.lib.progress import KrakenProgressBar
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.default_specs import SEGMENTATION_HYPER_PARAMS, SEGMENTATION_SPEC

from kraken.ketos.util import _validate_manifests, _expand_gt, message, to_ptl_device

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2

def _validate_merging(ctx, param, value):
    """
    Maps baseline/region merging to a dict of merge structures.
    """
    if not value:
        return None
    merge_dict = {}  # type: Dict[str, str]
    try:
        for m in value:
            k, v = m.split(':')
            merge_dict[v] = k  # type: ignore
    except Exception:
        raise click.BadParameter('Mappings must be in format target:src')
    return merge_dict

@click.command('segtest')
@click.pass_context
@click.option('-m', '--model', show_default=True, type=click.Path(exists=True, readable=True),
              multiple=False, help='Model(s) to evaluate')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--workers', show_default=True, default=1, help='Number of OpenMP threads when running on CPU.')
@click.option('--force-binarization/--no-binarization', show_default=True,
              default=False, help='Forces input images to be binary, otherwise '
              'the appropriate color format will be auto-determined through the '
              'network specification. Will be ignored in `path` mode.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page']), default='xml',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information.')
@click.option('--suppress-regions/--no-suppress-regions', show_default=True,
              default=False, help='Disables region segmentation training.')
@click.option('--suppress-baselines/--no-suppress-baselines', show_default=True,
              default=False, help='Disables baseline segmentation training.')
@click.option('-vr', '--valid-regions', show_default=True, default=None, multiple=True,
              help='Valid region types in training data. May be used multiple times.')
@click.option('-vb', '--valid-baselines', show_default=True, default=None, multiple=True,
              help='Valid baseline types in training data. May be used multiple times.')
@click.option('-mr',
              '--merge-regions',
              show_default=True,
              default=None,
              help='Region merge mapping. One or more mappings of the form `$target:$src` where $src is merged into $target.',
              multiple=True,
              callback=_validate_merging)
@click.option('-mb',
              '--merge-baselines',
              show_default=True,
              default=None,
              help='Baseline type merge mapping. Same syntax as `--merge-regions`',
              multiple=True,
              callback=_validate_merging)
@click.option('-br', '--bounding-regions', show_default=True, default=None, multiple=True,
              help='Regions treated as boundaries for polygonization purposes. May be used multiple times.')
@click.option("--threshold", type=click.FloatRange(.01, .99), default=.3, show_default=True,
              help="Threshold for heatmap binarization. Training threshold is .3, prediction is .5")
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def segtest(ctx, model, evaluation_files, device, workers, threshold,
            force_binarization, format_type, test_set, suppress_regions,
            suppress_baselines, valid_regions, valid_baselines, merge_regions,
            merge_baselines, bounding_regions):
    """
    Evaluate on a test set.
    """
    if not model:
        raise click.UsageError('No model to evaluate given.')

    from torch.utils.data import DataLoader
    import torch
    import torch.nn.functional as F

    from kraken.lib.train import BaselineSet, ImageInputTransforms
    from kraken.lib.vgsl import TorchVGSLModel

    logger.info('Building test set from {} documents'.format(len(test_set) + len(evaluation_files)))

    message('Loading model {}\t'.format(model), nl=False)
    nn = TorchVGSLModel.load_model(model)
    message('\u2713', fg='green')

    test_set = list(test_set)
    if evaluation_files:
        test_set.extend(evaluation_files)

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    _batch, _channels, _height, _width = nn.input
    transforms = ImageInputTransforms(
        _batch,
        _height, _width, _channels, 0,
        valid_norm=False, force_binarization=force_binarization
    )
    if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
        logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
        torch.multiprocessing.set_sharing_strategy('file_system')

    if not valid_regions:
        valid_regions = None
    if not valid_baselines:
        valid_baselines = None

    if suppress_regions:
        valid_regions = []
        merge_regions = None
    if suppress_baselines:
        valid_baselines = []
        merge_baselines = None

    test_set = BaselineSet(test_set,
                           line_width=nn.user_metadata["hyper_params"]["line_width"],
                           im_transforms=transforms,
                           mode=format_type,
                           augmentation=False,
                           valid_baselines=valid_baselines,
                           merge_baselines=merge_baselines,
                           valid_regions=valid_regions,
                           merge_regions=merge_regions)

    test_set.class_mapping = nn.user_metadata["class_mapping"]
    test_set.num_classes = sum([len(classDict) for classDict in test_set.class_mapping.values()])

    baselines_diff = set(test_set.class_stats["baselines"].keys()).difference(test_set.class_mapping["baselines"].keys())
    regions_diff = set(test_set.class_stats["regions"].keys()).difference(test_set.class_mapping["regions"].keys())

    if baselines_diff:
        message(f'Model baseline types missing in test set: {", ".join(sorted(list(baselines_diff)))}')

    if regions_diff:
        message(f'Model region types missing in the test set: {", ".join(sorted(list(regions_diff)))}')

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    ds_loader = DataLoader(test_set, batch_size=1, num_workers=workers, pin_memory=True)

    device = "cpu"
    nn.to(device)
    nn.eval()
    nn.set_num_threads(1)
    pages = []

    lines_idx = list(test_set.class_mapping["baselines"].values())
    regions_idx = list(test_set.class_mapping["regions"].values())

    all_preds = []
    all_targets = []

    # For storing all predictions and targets
    all_preds_map = []
    all_targets_map = []

    with KrakenProgressBar() as progress:
        batches = len(ds_loader)
        pred_task = progress.add_task('Evaluating', total=batches, visible=True)

        # for each document
        for batch in ds_loader:
            x, y = batch['image'], batch['target']
            try:
                pred, _ = nn.nn(x)
                # scale target to output size
                y = F.interpolate(y, size=(pred.size(2), pred.size(3))).squeeze(0).bool()
                pred = pred.squeeze()

                # Flatten predictions and targets
                all_preds.extend(pred[lines_idx].view(-1).detach().cpu().numpy())
                all_targets.extend(y[lines_idx].view(-1).detach().cpu().numpy())

                pred = pred > threshold
                pred = pred.view(pred.size(0), -1)
                y = y.view(y.size(0), -1)
                pages.append({
                    'intersections': (y & pred).sum(dim=1, dtype=torch.double),
                    'unions': (y | pred).sum(dim=1, dtype=torch.double),
                    'corrects': torch.eq(y, pred).sum(dim=1, dtype=torch.double),
                    'cls_cnt': y.sum(dim=1, dtype=torch.double),
                    'all_n': torch.tensor(y.size(1), dtype=torch.double, device=device)
                })
                if lines_idx:
                    y_baselines = y[lines_idx].sum(dim=0, dtype=torch.bool)
                    pred_baselines = pred[lines_idx].sum(dim=0, dtype=torch.bool)
                    pages[-1]["baselines"] = {
                        'intersections': (y_baselines & pred_baselines).sum(dim=0, dtype=torch.double),
                        'unions': (y_baselines | pred_baselines).sum(dim=0, dtype=torch.double),
                        'b_percent': torch.eq(y[lines_idx], pred[lines_idx]).sum(dim=1, dtype=torch.double).numpy()[0] /
                        (torch.eq(y[lines_idx], pred[lines_idx]).sum(dim=1, dtype=torch.double).numpy()[0] + torch.ne(y[lines_idx], pred[lines_idx]).sum(dim=1, dtype=torch.double).numpy()[0]),
                    }
                if regions_idx:
                    y_regions_idx = y[regions_idx].sum(dim=0, dtype=torch.bool)
                    pred_regions_idx = pred[regions_idx].sum(dim=0, dtype=torch.bool)
                    pages[-1]["regions"] = {
                        'intersections': (y_regions_idx & pred_regions_idx).sum(dim=0, dtype=torch.double),
                        'unions': (y_regions_idx | pred_regions_idx).sum(dim=0, dtype=torch.double),
                    }
                
                # Store predictions and targets
                all_preds_map.append(pred[lines_idx].cpu().numpy())
                all_targets_map.append(y[lines_idx].cpu().numpy())


            except FileNotFoundError as e:
                batches -= 1
                progress.update(pred_task, total=batches)
                logger.warning('{} {}. Skipping.'.format(e.strerror, e.filename))
            except KrakenInputException as e:
                batches -= 1
                progress.update(pred_task, total=batches)
                logger.warning(str(e))
            progress.update(pred_task, advance=1)

    # Accuracy / pixel
    corrects = torch.stack([x['corrects'] for x in pages], -1).sum(dim=-1)
    all_n = torch.stack([x['all_n'] for x in pages]).sum()  # Number of pixel for all pages

    class_pixel_accuracy = corrects / all_n
    mean_accuracy = torch.mean(class_pixel_accuracy)

    intersections = torch.stack([x['intersections'] for x in pages], -1).sum(dim=-1)
    unions = torch.stack([x['unions'] for x in pages], -1).sum(dim=-1)
    smooth = torch.finfo(torch.float).eps
    class_iu = (intersections + smooth) / (unions + smooth)
    mean_iu = torch.mean(class_iu)

    cls_cnt = torch.stack([x['cls_cnt'] for x in pages]).sum()
    freq_iu = torch.sum(cls_cnt / cls_cnt.sum() * class_iu.sum())

    message(f"Mean Accuracy: {mean_accuracy.item():.3f}")
    message(f"Mean IOU: {mean_iu.item():.3f}")
    message(f"Frequency-weighted IOU: {freq_iu.item():.3f}")

    # Region accuracies
    if lines_idx:
        line_intersections = torch.stack([x["baselines"]['intersections'] for x in pages]).sum()
        line_unions = torch.stack([x["baselines"]['unions'] for x in pages]).sum()
        smooth = torch.finfo(torch.float).eps
        line_iu = (line_intersections + smooth) / (line_unions + smooth)
        message(f"Class-independent Baseline IOU: {line_iu.item():.3f}")
        b_percent = math.fsum([x["baselines"]['b_percent'] for x in pages]) / len(pages)
        message(f"Percentage of correct Baselines: {b_percent:.3f}")

    # Region accuracies
    if regions_idx:
        region_intersections = torch.stack([x["regions"]['intersections'] for x in pages]).sum()
        region_unions = torch.stack([x["regions"]['unions'] for x in pages]).sum()
        smooth = torch.finfo(torch.float).eps
        region_iu = (region_intersections + smooth) / (region_unions + smooth)
        message(f"Class-independent Region IOU: {region_iu.item():.3f}")

    # Calculate mAP@50
    #all_preds_map = np.concatenate(all_preds_map, axis=1)
    #all_targets_map = np.concatenate(all_targets_map, axis=1)
    #aps = []
    #for i in range(all_preds_map.shape[0]):
    #    ap = average_precision_score(all_targets_map[i], all_preds_map[i])
    #    aps.append(ap)
    #map_50 = np.mean(aps)
    #message(f"mAP@50: {map_50:.3f}")

    # Calculate Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    pr_auc = auc(recall, precision)

    # Compute mean AP over recall range
    indices = np.where(recall[:-1] != recall[1:])[0] + 1
    mAP = np.sum((recall[indices] - recall[indices - 1]) *
                 precision[indices])
    message(f"mAP@50: {abs(mAP):.3f}")

    # Plot Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision, color='b', label=f'PR curve (area = {pr_auc:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    from rich.console import Console
    from rich.table import Table

    table = Table('Category', 'Class Name', 'Pixel Accuracy', 'IOU', 'Object Count')

    class_iu = class_iu.tolist()
    class_pixel_accuracy = class_pixel_accuracy.tolist()
    for (cat, class_name), iu, pix_acc in zip(
        [(cat, key) for (cat, subcategory) in test_set.class_mapping.items() for key in subcategory],
        class_iu,
        class_pixel_accuracy
    ):
        table.add_row(cat, class_name, f'{pix_acc:.3f}', f'{iu:.3f}', f'{test_set.class_stats[cat][class_name]}' if cat != "aux" else 'N/A')

    console = Console()
    console.print(table)

if __name__ == '__main__':
    segtest()
