############################
# CMAKE VT100 Escape Codes
# Symbology:
#		\33		= escape character (ascii 27 decimal)
#		<v>		= single or double digit number.  Vertical coordinate
#		<h>		= single or double digit number.  Horizontal coordinate
#		<n>		= single or double digit number.  Number of chars/lines
#		others   = single characters just as they appear.
# NOTE: Many sequences have "\33[" which is two chars: "escape" and "[".
############################
set(VT100_PRE      "\\033[")
set(VT100_POST      "m")

set(VT100_RESET      "${VT100_PRE}0${VT100_POST}")
set(VT100_BOLD       "${VT100_PRE}1${VT100_POST}")
set(VT100_DIM        "${VT100_PRE}2${VT100_POST}")
set(VT100_UNDERLINE  "${VT100_PRE}4${VT100_POST}")
set(VT100_BLINK      "${VT100_PRE}5${VT100_POST}")
set(VT100_REVERSE    "${VT100_PRE}7${VT100_POST}")
set(VT100_HIDDEN     "${VT100_PRE}8${VT100_POST}")
set(VT100_CLEARLINE  "${VT100_PRE}K")

set(VT100_FG_BLACK   "${VT100_PRE}30${VT100_POST}")
set(VT100_FG_RED     "${VT100_PRE}31${VT100_POST}")
set(VT100_FG_GREEN   "${VT100_PRE}32${VT100_POST}")
set(VT100_FG_YELLOW  "${VT100_PRE}33${VT100_POST}")
set(VT100_FG_BLUE    "${VT100_PRE}34${VT100_POST}")
set(VT100_FG_MAGENTA "${VT100_PRE}35${VT100_POST}")
set(VT100_FG_CYAN    "${VT100_PRE}36${VT100_POST}")
set(VT100_FG_WHITE   "${VT100_PRE}37${VT100_POST}")

set(VT100_BG_BLACK   "${VT100_PRE}40${VT100_POST}")
set(VT100_BG_RED     "${VT100_PRE}41${VT100_POST}")
set(VT100_BG_GREEN   "${VT100_PRE}42${VT100_POST}")
set(VT100_BG_YELLOW  "${VT100_PRE}43${VT100_POST}")
set(VT100_BG_BLUE    "${VT100_PRE}44${VT100_POST}")
set(VT100_BG_MAGENTA "${VT100_PRE}45${VT100_POST}")
set(VT100_BG_CYAN    "${VT100_PRE}46${VT100_POST}")
set(VT100_BG_WHITE   "${VT100_PRE}47${VT100_POST}")

set(VT100_getcursor	"\\336n") 			# Get cursor position
set(VT100_respcursor	"\\33<v>;<h>R") 	# Response: cursor is at v,h


#################
# INFO function #
#################
FUNCTION(info msg)
execute_process(COMMAND /bin/echo -e "--" ${msg} ${VT100_RESET})
ENDFUNCTION(info)
