from ascii_magic import AsciiArt

my_art = AsciiArt.from_image('eva.jpg')
my_art.to_html_file('ascii_art.html', columns=1000, width_ratio=2)