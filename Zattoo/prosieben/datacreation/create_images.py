from PIL import Image, ImageChops, ImageOps, ImageFilter


def plane_background(width_logo=True):
    imgs = []
    padding = 9
    borders = [
        (padding, padding, padding, padding),
        (padding - 1, padding, padding + 1, padding),
        (padding + 1, padding, padding - 1, padding),
        (padding, padding + 1, padding, padding - 1),
        (padding, padding - 1, padding, padding + 1),
        (padding + 1, padding + 1, padding - 1, padding - 1),
        (padding - 1, padding - 1, padding + 1, padding + 1),
        (padding - 1, padding + 1, padding + 1, padding - 1),
        (padding + 1, padding - 1, padding - 1, padding + 1),
    ]
    filters = [
        ImageFilter.DETAIL,
        ImageFilter.SMOOTH,
        ImageFilter.SMOOTH_MORE,
        ImageFilter.SHARPEN,
    ]
    logo = Image.open("C:\PyCharm\WerbeSkip\Zattoo\prosieben\datacreation\logo32x32.png")
    logo = logo.convert(mode="L")  # mode L is white and black
    for border in borders:
        exp_logo = ImageOps.expand(logo, border, fill="black")
        for color in range(0, 221):
            img = Image.new("L", color=color, size=exp_logo.size)
            if width_logo:
                img = ImageChops.screen(exp_logo, img)
            imgs.append(img.copy())
            for filter in filters:
                img_filter = img.filter(filter=filter)
                imgs.append(img_filter.copy())
    return imgs


if __name__ == "__main__":
    print(len(plane_background(width_logo=True)))
    plane_background(width_logo=True)[230 * 5].show()
