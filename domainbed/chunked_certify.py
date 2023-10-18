from utils.deform_smooth import SmoothFlow
from time import time
import torch
import datetime
import os
import math



def chunked_certify(base_classifier, test_dataset, args):
    """
    Perform certification of a given base classifier using an interval (chunk) of a test dataset.

    Args:
        base_classifier (torch.nn.Module): The base classifier to be certified.
        test_dataset (torch.utils.data.Dataset): The dataset used for certification.
        args (argparse.Namespace): A namespace containing various arguments and configurations.

    Returns:
        None (The certification results are saved to a file).

    Note:
        The certification results are saved in a file specified by args.outfile.

    """
    
    # Set up the directory for storing certification results based on target_domain flag
    if not args.target_domain:
        args.basedir = os.path.join(args.output_dir, 'source/certify/')
    else:
        args.basedir = os.path.join(args.output_dir, 'target/certify/')

    # Set the value of sigma for smoothing
    if args.sigma != 0:
        sigma = args.sigma
    else:
        sigma = 0.1

    if args.certify_method == 'rotation':
        sigma *= math.pi # For rotaions to transform the angles to [0, pi]

    # Create the output directory if it does not exist
    if not os.path.exists(args.basedir):
        os.makedirs(args.basedir, exist_ok=True)

    # Set the output file for storing certification results
    args.outfile = os.path.join(args.basedir, f'sigma_{args.sigma}.txt')

    # Create the smoothed classifier g using the SmoothFlow class
    smoothed_classifier = SmoothFlow(base_classifier, args.num_classes, args.certify_method, sigma)

    # Use uniform smoothing for rotation and scaling
    if args.certify_method in ['rotation', 'scaling_uniform']:
        args.uniform = True
        
    # Prepare the output file for storing certification results
    if os.path.exists(args.outfile):
        f = open(args.outfile, 'a')
    else:
        f = open(args.outfile, 'w')
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # Compute the interval size for chunked certification based on the number of chunks
    interval = args.max//args.chunks
    start_ind = args.num_chunk * interval

    print(interval)
    print(start_ind)

    # Iterate through the chunk to certify examples
    for i in range(start_ind, start_ind + interval):

        # only certify every args.skip examples, and stop after max_samples
        if i % args.skip != 0:
            continue

        # Retrieve the input and label for the current example from the dataset
        (x,label) = test_dataset[i]
        label = torch.tensor([label])
        x, label = x.squeeze(), label.squeeze()
        before_time = time()
        
        # Certify the prediction of the smoothed classifier around input x
        x = x.cuda()
        prediction, radius, p_A = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.certify_batch_sz)

        # Adjust the radius using the uniform smoothing approach if necessary
        if args.uniform:
            radius = 2 * sigma * (p_A - 0.5)
        after_time = time()

        # Determine if the prediction is correct
        correct = int(prediction == label)

        # Compute the time taken for certifying one image
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # Print and save the certification results for the current example
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
    print("Certification is done")