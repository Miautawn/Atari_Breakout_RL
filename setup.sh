# install x-server and unrar
sudo apt install xvfb unrar

# get the roms and import them via atari_py
wget http://www.atarimania.com/roms/Roms.rar
unrar e Roms.rar ./ROMS/

#import the roms via ale-py
ale-import-roms ./ROMS/

#remove the roms.rar
rm Roms.rar
