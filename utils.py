def depth_guided_class_specific_consistency_loss(logits, depth_map, label, ignore_mask, threshold=0.95, margin=0.15, ignore_value=255):
    """
    Class-specific depth-guided consistency loss using weakly-augmented predictions (label).
    
    Args:
        logits: Logits from the model (B, num_classes, H, W)
        depth_map: Depth map (B, 1, H, W)
        label: Weakly-augmented prediction labels (B, H, W)
        ignore_mask: Ignore mask for regions that shouldn't be considered in the loss
        threshold: Confidence threshold for defining low-confidence regions
        margin: Tolerance margin for depth difference in consistency loss
        ignore_value: Value in ignore_mask indicating regions to ignore in the loss calculation
    
    Returns:
        loss: Consistency loss value based on class-specific low-confidence regions
    """
    B, num_classes, H, W = logits.shape

    # Downscale depth map and ignore mask to match the resolution of the logits
    depth_map = F.interpolate(depth_map.unsqueeze(1), size=(H // 10, W // 10), mode='bilinear', align_corners=True)
    ignore_mask = F.interpolate(ignore_mask.unsqueeze(1).float(), size=(H // 10, W // 10), mode="bilinear").squeeze(1)

    # Downscale logits and labels for consistency with the reduced resolution
    logits = F.interpolate(logits, size=(H // 10, W // 10), mode='bilinear', align_corners=True)

    label = F.interpolate(label.unsqueeze(1).float(), size=(H // 10, W // 10), mode="bilinear").squeeze(1).long()
    
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Identify unique classes in the current batch
    unique_classes = label.unique()
    total_loss = 0.0
    count = 0

    indices = torch.randperm(unique_classes.numel())
    unique_classes_shuffle = unique_classes[indices] # 将类别序列打乱

    prev_class_logits = None
    for c in unique_classes_shuffle:
        if c == ignore_value:
            continue  # Skip ignore class

        # Mask for pixels predicted as class `c` in the weakly-augmented label
        class_mask = (label == c).float()  # Only pixels where the weak prediction is class `c`

        # Define low-confidence regions within this class, excluding ignored areas
        low_confidence_mask = ((probabilities[:, c, :, :] < threshold) & (ignore_mask != ignore_value)).float()
        class_low_conf_mask = low_confidence_mask * class_mask  # Only low-confidence pixels within class `c`

        if class_low_conf_mask.sum() == 0:
            # Skip if no low-confidence pixels in the current class
            continue


        # Flatten depth map and low-confidence mask for matrix calculations
        depth_flat = depth_map.view(B, -1)  # (B, H*W)
        logits_flat = logits.view(B, num_classes, -1)  # (B, num_classes, H*W)
        class_low_conf_mask_flat = class_low_conf_mask.view(B, -1)  # Flatten (B, H*W)
        
        logits_flat_clone = logits_flat.permute(1,0,2)
        current_low_pixel_logit = logits_flat_clone[:,class_low_conf_mask_flat.type(torch.bool)]  # num_classes * N
        current_low_pixel_reverse = current_low_pixel_logit.permute(1,0)  #  N * num_classes
        if prev_class_logits is not None:
            logit_simialarity = torch.matmul(current_low_pixel_reverse, prev_class_logits.permute(1,0))
            loss_intra_class = torch.mean(torch.abs(torch.sigmoid(logit_simialarity))) / class_low_conf_mask.sum()

        # Calculate depth differences between each pair of pixels
        depth_diff_matrix = torch.abs(depth_flat.unsqueeze(2) - depth_flat.unsqueeze(1))  # (B, H*W, H*W)

        # Compute logit similarity matrix for class `c`
        logits_class_flat = logits_flat[:, c, :]  # (B, H*W)
        logit_similarity_matrix = torch.matmul(logits_class_flat.unsqueeze(2), logits_class_flat.unsqueeze(1))  # (B, H*W, H*W)
        logit_similarity_matrix = torch.sigmoid(logit_similarity_matrix)  # Optional: Apply sigmoid to keep values in [0, 1]
        
        # Depth-guided mask for regions with similar depth
        consistency_mask = (depth_diff_matrix < margin).float()  # Mask of pixels with similar depth (B, H*W, H*W)
        
        # Mask for low-confidence regions within class `c` with similar depth
        low_confidence_region_mask = class_low_conf_mask_flat.unsqueeze(2) * class_low_conf_mask_flat.unsqueeze(1)  # (B, H*W, H*W)
        combined_mask = low_confidence_region_mask * consistency_mask
        combined_mask = torch.clamp(combined_mask, min=1e-6)  # Avoid zero values

        # Calculate consistency loss for class `c`
        class_loss = torch.mean(combined_mask * torch.abs(1 - logit_similarity_matrix))
        
        # Accumulate loss and count for averaging
        total_loss += class_loss
        count += 1

        if prev_class_logits is not None:
            total_loss = total_loss + loss_intra_class

        prev_class_logits =  current_low_pixel_reverse


    # Average loss over all classes
    if count > 0:
        total_loss /= count

    

    return total_loss