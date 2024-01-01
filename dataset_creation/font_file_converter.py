import fontforge
import os

#* ASCII HTML Numbers
#* Numbers   48-57
#* Uppercase 65-90
#* Lowercase 97-122

glyphName = tuple(  [str(i) + "_num" for i in range(10)] +          #* Numbers 0-9
                    [chr(i) + "_upper" for i in range(97, 123)] +   #* Uppercase letters A-Z
                    [chr(i) + "_lower" for i in range(97, 123)])    #* Lowercase letters a-z

#* Font character map; 48-57 = numbers [0-9], 65-90 = uppercase letters [A-Z], 97-122 = lowercase letters [a-z]

glyphNum = tuple(list(range(48, 58)) +      #* Numbers 0-9
                 list(range(65, 91)) +      #* Uppercase letters A-Z
                 list(range(97, 123)))      #* Lowercase letters a-z

def export_png_from_font(file, fontDir, targetDir):

    print(fontDir + '/' + file)
    try:
        ff = fontforge.open(fontDir + '/' + file)
    except:
        return
    fontName = file.replace(' ', '_').split('.')[0]

    for i in range(len(glyphName)):
        pngName = targetDir + '/' + glyphName[i] + '_' + fontName + '.png'
        try:
            ff[glyphNum[i]].export(pngName)
        except:
            pass

def export_all_fonts(fontDir, targetDir):
    
    dirFiles = os.listdir(fontDir)
    fontFiles = []

    for f in dirFiles:
        if f.lower().endswith(('.ttf', 'ttc', '.otf', 'fnt', 'bdf', 'fon', 'woff')):
            fontFiles.append(f)

    for f in fontFiles:
        print("===!===" + f + "===!===")
        export_png_from_font(f, fontDir, targetDir)
