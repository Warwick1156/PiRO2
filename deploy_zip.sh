#!bash

package_name=130759_131846

cd deploy
zip $package_name.zip $package_name.pdf emails.txt environment.yml run.py
mv $package_name.zip ../
