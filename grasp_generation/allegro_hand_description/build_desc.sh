
if [ "$1" == "-h" -o "$2" == "-h" ]
then
 echo -e "\n\n      BUILD_DESC.SH\n"
 echo -e "   INSTRUCTIONS FOR USE\n   --------------------\n"
 echo -e "'build_desc.sh' can be called with up to two (2) arguments.\nThe first argument is the side of the hand, left or right, and\nthe second argument determines the use of a robot tree diagram.\n\n"
 echo -e "Examples:\n"
 echo -e "   $ bash build_desc.sh RIGHT 1\n   Builds the right hand URDF from its xacro file and shows the robot tree diagram when finished.\n"
 echo -e "   $ bash build_desc.sh left\n   Builds the left hand URDF from its xacro file and DOES NOT show the tree diagram.\n\n\n"
 echo -e "Possible variations of the 1st argument are:\n RIGHT, Right, right, R, r, LEFT, Left, left, L and l.\n"
 echo -e "For the 2nd argument, '1' produces the tree graph.\nLeft blank or passed any other value, the graph will not be created."
 echo -e "\n\n"
 exit 0
fi
	

if [ "$1" == "right" -o "$1" == "Right" -o "$1" == "RIGHT" -o "$1" == "R" -o "$1" == "r" ]
then
	HAND="RIGHT"
	hand="right"
elif [ "$1" == "left" -o "$1" == "Left" -o "$1" == "LEFT" -o "$1" == "L" -o "$1" == "l" ]
then
	HAND="LEFT"
	hand="left"
else
 echo -e "\n\nWrong or incorrect amount of arguments.\nUse 'bash build_desc.sh -h' for instructions.\n\n"
 echo -e "If you see an error like\n'[: 7: right: unexpected operator',\nuse 'bash build_desc.sh' instead of 'sh build_desc.sh'.\n\n"
 exit 0
fi


echo -e "\n\nBuilding the [[$HAND]] Allegro Hand URDF...\n"


rosrun xacro xacro.py allegro_hand_description_$hand.xacro -o allegro_hand_description_$hand.urdf

rosrun urdf check_urdf allegro_hand_description_$hand.urdf 

echo -e "\n"





if [ "$2" == "1" ] 
then
	rosrun urdf_parser urdf_to_graphiz allegro_hand_description_$hand.urdf 
	rm *.gv
	echo -e "Removed file allegro_hand_$hand.gv\n"
	echo -e "Opening graph pdf...\n\n"
	gnome-open allegro_hand_$hand.pdf
else
	echo -e "NOTE:\nTo open a graph of the hand upon URDF compilation,\nenter "1" as the second arg when calling 'bash build.sh'\n\n"
fi
