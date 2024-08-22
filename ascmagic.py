from ascii_magic import AsciiArt


my_art = AsciiArt.from_image(r'C:\Users\marko\PycharmProjects\smalltestv2\kaij\kaij (2).jpg')
my_art.to_html_file('ascii_art2.html', columns=600, width_ratio=1)