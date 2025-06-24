import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import nibabel as nib
import os
import SimpleITK as sitk


#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
train_folder = './Output/emidec_dataset/train/'
Bulls_eye_plots_path_Vol = './Output/emidec_dataset/train_vol_plots'
template_img_path = "./Template/myo_ED_AHA17.nii.gz"
MM_TO_ML_FACTOR = 0.001
#---------------------------------------------------------------------------------------------------------------------------------------

#Getting all images from folders
def get_all_images(folder, ext):
    all_files = []
    for file in os.listdir(folder):
        _,  file_ext = os.path.splitext(file)
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)
    return all_files

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def main():
    train_files = []
    for folder_path in os.listdir(train_folder):
        if('Case_P' in folder_path):
            mask_file_path = os.path.join(train_folder, folder_path + '/Contours')
            train_files.append(get_all_images(mask_file_path, 'gz'))
        else:
            continue
    train_masks = []
    for i in range(len(train_files)):
        gd = [x for x in train_files[i] if 'reg2' in x]
        train_masks.append(gd[0])
    train_masks.sort()
    #--------------------------------------------------------------------------------
    fit_average_bulleye_plot1(train_masks, template_img_path)
    exit()
    for i in range(len(train_masks)):
        fit_average_bulleye_plot([train_masks[i]], template_img_path, Bulls_eye_plots_path_Vol)
        exit()

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def fit_bulleye_plot(subject_img_path, template_img_path):
    sub_img = nib.load(subject_img_path)
    temp_img = nib.load(template_img_path)
    sub_img_data = sub_img.get_fdata()
    temp_img_data = temp_img.get_fdata()

    sub_img_data[sub_img_data == 1] = 0
    sub_img_data[sub_img_data == 2] = 0
    sub_img_data[sub_img_data == 3] = 1
    sub_img_data[sub_img_data == 4] = 0
    aha = np.unique(temp_img_data)

    aha_17 = []
    for i in range(len(aha)):
        if(i==0):
            continue
        else:
            atlas = np.zeros((temp_img_data.shape[0], temp_img_data.shape[1], temp_img_data.shape[2], 1))
            atlas[temp_img_data == aha[i]] = 1
            atlas = np.resize(atlas, (temp_img_data.shape[0], temp_img_data.shape[1], temp_img_data.shape[2]))
            out = sub_img_data * atlas

            # Compute volume statistics
            image_spacing = sitk.ReadImage(subject_img_path).GetSpacing()
            pixelVolume = image_spacing[0] * image_spacing[1] * image_spacing[2]
            num_Pixels = np.count_nonzero(out == 1)
            img_Volume = num_Pixels * pixelVolume * MM_TO_ML_FACTOR
            aha_17.append(img_Volume)

    print(aha_17)
    return aha_17

#-------------------------------------------------------------------------

def fit_average_bulleye_plot1(list_subject_img_path, template_img_path):
    avg_aha_17 = []
    for i in range(len(list_subject_img_path)):
        aha_17 = fit_bulleye_plot(list_subject_img_path[i], template_img_path)
        avg_aha_17.append(aha_17)

    avg_aha_17 = np.array(avg_aha_17)
    avg_aha_17 = np.average(avg_aha_17, axis=0)
    # print(avg_aha_17)
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1, subplot_kw=dict(projection='polar'))
    cmap = mpl.cm.Blues  #coolwarm, twilight
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(avg_aha_17))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label='Scar Volume (ML)')
    bullseye_plot(ax, avg_aha_17, cmap=cmap, norm=norm, labels=labels, labelProps={'size':12, 'color':'gray'})
    ax.set_title('Average Bulls Eye (AHA) Plot')
    plt.show()

#-------------------------------------------------------------------------

def fit_average_bulleye_plot(list_subject_img_path, template_img_path, Bulls_eye_plots_path):
    avg_aha_17_sub = []
    for i in range(len(list_subject_img_path)):
        aha_17_sub = fit_bulleye_plot(list_subject_img_path[i], template_img_path)
        avg_aha_17_sub.append(aha_17_sub)
    avg_aha_17_sub = np.array(avg_aha_17_sub)
    avg_aha_17_sub = np.average(avg_aha_17_sub, axis=0)

    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    fig, ax = plt.subplots(figsize=(12, 12), nrows=1, ncols=1, subplot_kw=dict(projection='polar'))
    cmap = mpl.cm.Blues  #coolwarm, twilight
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(avg_aha_17_sub))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label='Scar Volume (ML)')
    plot_name = list_subject_img_path[0].split('/')[-3]
    ax.set_title('Bulls Eye (AHA) Plot: ' + plot_name)
    bullseye_plot(ax, avg_aha_17_sub, cmap=cmap, norm=norm, labels=labels, labelProps={'size': 12, 'color': 'gray'})
    # plt.show()

    # Save plots
    if not os.path.exists(Bulls_eye_plots_path):
        os.makedirs(Bulls_eye_plots_path)
    plot_name = list_subject_img_path[0].split('/')[-1].split('.')[-3]
    plot_name = Bulls_eye_plots_path + '/' + plot_name.replace('_2', '') + '.png'
    fig.savefig(plot_name)

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def bullseye_plot(ax, data, seg_bold=None, cmap="viridis", norm=None, labels=[], labelProps={}):
    """
    Bullseye representation for the left ventricle.

    Parameters
    ----------
    ax : Axes
    data : list[float]
        The intensity values for each of the 17 segments.
    seg_bold : list[int], optional
        A list with the segments to highlight.
    cmap : colormap, default: "viridis"
        Colormap for the data.
    norm : Normalize or None, optional
        Normalizer for the data.

    Notes
    -----
    This function creates the 17 segment model for the left ventricle according
    to the American Heart Association (AHA) [1]_

    References
    ----------
    .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
        S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
        and M. S. Verani, "Standardized myocardial segmentation and
        nomenclature for tomographic imaging of the heart",
        Circulation, vol. 105, no. 4, pp. 539-542, 2002.
    """

    data = np.ravel(data)
    if seg_bold is None:
        seg_bold = []
    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    r = np.linspace(0.2, 1, 4)

    ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
    ax.grid(False)  # Remove grid

    # Fill segments 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
            (16, 17, 0, r[0]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               color=cmap(norm(data[start:stop])))

    # Now, draw the segment borders.  In order for the outer bold borders not
    # to be covered by inner segments, the borders are all drawn separately
    # after the segments have all been filled.  We also disable clipping, which
    # would otherwise affect the outermost segment edges.
    # Draw edges of segments 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               clip_on=False, color="none", edgecolor="k", linewidth=[
                   4 if i + 1 in seg_bold else 2 for i in range(start, stop)])

    # Draw edge of segment 17 -- here; the edge needs to be drawn differently,
    # using plot().
    ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "k",
            linewidth=(4 if 17 in seg_bold else 2))

    if labels:
        ax.annotate(labels[0], xy=(dtheta + 0 * np.pi / 180, np.mean(r[2:4])), ha='center', va='center', **labelProps)
        ax.annotate(labels[1], xy=(dtheta + 60 * np.pi / 180, np.mean(r[2:4])), ha='center', va='center', **labelProps)
        ax.annotate(labels[2], xy=(dtheta + 120 * np.pi / 180, np.mean(r[2:4])), ha='center', va='center', **labelProps)
        ax.annotate(labels[3], xy=(dtheta + 180 * np.pi / 180, np.mean(r[2:4])), ha='center', va='center', **labelProps)
        ax.annotate(labels[4], xy=(dtheta + 240 * np.pi / 180, np.mean(r[2:4])), ha='center', va='center', **labelProps)
        ax.annotate(labels[5], xy=(dtheta + 300 * np.pi / 180, np.mean(r[2:4])), ha='center', va='center', **labelProps)
        ax.annotate(labels[6], xy=(dtheta + 0 * np.pi / 180, np.mean(r[1:3])), ha='center', va='center', **labelProps)
        ax.annotate(labels[7], xy=(dtheta + 60 * np.pi / 180, np.mean(r[1:3])), ha='center', va='center', **labelProps)
        ax.annotate(labels[8], xy=(dtheta + 120 * np.pi / 180, np.mean(r[1:3])), ha='center', va='center', **labelProps)
        ax.annotate(labels[9], xy=(dtheta + 180 * np.pi / 180, np.mean(r[1:3])), ha='center', va='center', **labelProps)
        ax.annotate(labels[10], xy=(dtheta + 240 * np.pi / 180, np.mean(r[1:3])), ha='center', va='center', **labelProps)
        ax.annotate(labels[11], xy=(dtheta + 300 * np.pi / 180, np.mean(r[1:3])), ha='center', va='center', **labelProps)
        ax.annotate(labels[12], xy=(dtheta + 0 * np.pi / 180, np.mean(r[0:2])), ha='center', va='center', **labelProps)
        ax.annotate(labels[13], xy=(dtheta + 90 * np.pi / 180, np.mean(r[0:2])), ha='center', va='center', **labelProps)
        ax.annotate(labels[14], xy=(dtheta + 180 * np.pi / 180, np.mean(r[0:2])), ha='center', va='center', **labelProps)
        ax.annotate(labels[15], xy=(dtheta + 270 * np.pi / 180, np.mean(r[0:2])), ha='center', va='center', **labelProps)
        ax.annotate(labels[16], xy=(dtheta + 0 * np.pi / 180, np.mean([0.0])), ha='center', va='center', **labelProps)

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()