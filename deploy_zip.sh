#!bash

package_name=130759_131846

cd deploy
zip -r $package_name.zip *
mv $package_name.zip ../
