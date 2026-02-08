# Commands
# python stopsign.py --show-image images/road62.png
#python stopsign.py --detect images/road62.png
#python stopsign.py --detectall images

import argparse
import os
import time
import xml.etree.ElementTree as ET

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path: str) -> np.ndarray:
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def load_template(template_path: str, size: tuple[int, int] | None = (64, 64)) -> np.ndarray:
    template = cv.imread(template_path)
    if template is None:
        raise FileNotFoundError(f"Could not read template: {template_path}")
    if size is not None:
        template = cv.resize(template, size)
    return template


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def detect_stop_sign(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.5,
    scales: np.ndarray | None = None,
) -> tuple[bool, list[int], float]:
    if scales is None:
        scales = np.linspace(0.5, 1.5, 11)

    image_gray = _to_gray_float(image)
    best_score = -1.0
    best_bbox = [0, 0, 1, 1]

    for scale in scales:
        scaled_w = int(template.shape[1] * scale)
        scaled_h = int(template.shape[0] * scale)
        if scaled_w < 8 or scaled_h < 8:
            continue
        if scaled_w > image.shape[1] or scaled_h > image.shape[0]:
            continue

        scaled_template = cv.resize(template, (scaled_w, scaled_h))
        template_gray = _to_gray_float(scaled_template)

        response = cv.matchTemplate(image_gray, template_gray, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(response)
        if max_val > best_score:
            best_score = max_val
            left, top = max_loc
            best_bbox = [top, left, scaled_h, scaled_w]

    detected = best_score >= threshold
    if not detected:
        best_bbox = [0, 0, 1, 1]

    return detected, best_bbox, best_score


def draw_bbox(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    image_copy = image.copy()
    top, left, height, width = bbox
    cv.rectangle(
        image_copy,
        (left, top),
        (left + width, top + height),
        (255, 0, 0),
        2,
    )
    return image_copy


def parse_annotation(annotation_path: str) -> tuple[bool, list[int] | None, str | None]:
    if not os.path.exists(annotation_path):
        return False, None, None

    tree = ET.parse(annotation_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        name = obj.findtext("name")
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = int(bbox.findtext("xmin"))
        ymin = int(bbox.findtext("ymin"))
        xmax = int(bbox.findtext("xmax"))
        ymax = int(bbox.findtext("ymax"))
        gt_bbox = [ymin, xmin, ymax - ymin, xmax - xmin]
        if name == "stop":
            return True, gt_bbox, name

    return False, None, None


def _default_paths() -> tuple[str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(base_dir, "data", "traffic-stop-signs", "template-1-2.png")
    annotations_dir = os.path.join(base_dir, "annotations")
    return template_path, annotations_dir


def _resolve_annotation_path(image_path: str, annotations_dir: str | None) -> str | None:
    if annotations_dir is None:
        return None
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(annotations_dir, f"{base}.xml")


def show_image_with_info(image_path: str, annotations_dir: str | None = None) -> None:
    image = load_image(image_path)
    annotation_path = _resolve_annotation_path(image_path, annotations_dir)
    gt_has_stop, gt_bbox, gt_name = parse_annotation(annotation_path) if annotation_path else (False, None, None)

    if gt_name is None:
        print("No annotation found for this image.")
    else:
        print(f"Traffic sign: {gt_name}")
        if gt_bbox is not None:
            print(f"Location: top={gt_bbox[0]}, left={gt_bbox[1]}, height={gt_bbox[2]}, width={gt_bbox[3]}")
            image = draw_bbox(image, gt_bbox)

    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


def detect_and_display(
    image_path: str,
    template_path: str,
    threshold: float,
    annotations_dir: str | None = None,
) -> None:
    image = load_image(image_path)
    template = load_template(template_path)
    detected, bbox, score = detect_stop_sign(image, template, threshold=threshold)

    name = "stop" if detected else "none"
    print(f"Detected: {'Yes' if detected else 'No'}")
    print(f"Name: {name}")
    if detected:
        print(f"Location: top={bbox[0]}, left={bbox[1]}, height={bbox[2]}, width={bbox[3]}")
        image = draw_bbox(image, bbox)

    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    label = f"Detected: {'Yes' if detected else 'No'} (score={score:.3f})"
    cv.putText(image_rgb, label, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


def detect_all(
    images_dir: str,
    template_path: str,
    threshold: float,
    annotations_dir: str | None,
) -> None:
    template = load_template(template_path)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
    image_files.sort()

    false_pos = 0
    false_neg = 0
    total = 0

    start = time.perf_counter()

    print("filename\tstop sign detected\tground truth")
    for filename in image_files:
        image_path = os.path.join(images_dir, filename)
        annotation_path = _resolve_annotation_path(image_path, annotations_dir)
        gt_has_stop, _, _ = parse_annotation(annotation_path) if annotation_path else (False, None, None)

        image = load_image(image_path)
        detected, _, _ = detect_stop_sign(image, template, threshold=threshold)

        detected_str = "Yes" if detected else "No"
        gt_str = "Yes" if gt_has_stop else "No"
        print(f"{filename}\t{detected_str}\t{gt_str}")

        if detected and not gt_has_stop:
            false_pos += 1
        if not detected and gt_has_stop:
            false_neg += 1
        total += 1

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print("")
    print(f"Total images processed\t{total}")
    print(f"False positives\t{false_pos}")
    print(f"False negatives\t{false_neg}")
    print(f"Total time taken (ms)\t{elapsed_ms:.2f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stop sign detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--show-image", dest="show_image", metavar="IMAGE", help="Display image and annotation")
    group.add_argument("--detect", dest="detect", metavar="IMAGE", help="Detect stop sign and display image")
    group.add_argument("--detectall", dest="detectall", metavar="DIR", help="Detect stop signs in folder of PNG images")

    parser.add_argument(
        "--template",
        default=None,
        help="Path to template image (default: lab1/data/traffic-stop-signs/template-1-2.png)",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Path to annotations folder (default: lab1/annotations)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    default_template, default_annotations = _default_paths()
    template_path = args.template or default_template
    annotations_dir = args.annotations or default_annotations

    if args.show_image:
        show_image_with_info(args.show_image, annotations_dir=annotations_dir)
        return

    if args.detect:
        detect_and_display(
            args.detect,
            template_path=template_path,
            threshold=args.threshold,
            annotations_dir=annotations_dir,
        )
        return

    if args.detectall:
        detect_all(
            args.detectall,
            template_path=template_path,
            threshold=args.threshold,
            annotations_dir=annotations_dir,
        )
        return


if __name__ == "__main__":
    main()
