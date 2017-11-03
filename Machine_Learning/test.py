from PIL import Image, ImageChops

im = Image.open("C:\PyCharm\WerbeSkip\Zattoo\images/161.jpg")
im2 = Image.open("C:\PyCharm\WerbeSkip\Zattoo\images/23.jpg")
im = ImageChops.screen(im2, im).convert("LA")
im.show()
print(list(im.getdata()))
