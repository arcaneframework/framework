#
# Find the 'Python' executable
#
# This module defines
# XERCESC_INCLUDE_DIR, where to find headers,
# XERCESC_LIBRARIES, the libraries to link against to use XercesC.
# PYTHONEXEC_FOUND, If false, do not try to use Python
 
FIND_PROGRAM(PYTHONEXEC_NAME NAMES python2.4 python
  PATHS
  /usr/local/opendev1/bin
  /usr/local/bin
  /usr/bin
  c:/utils/Python/2.4.2/bin
)
 
SET( PYTHONEXEC_FOUND "NO" )
IF(PYTHONEXEC_NAME)
  SET( PYTHONEXEC_FOUND "YES" )
ENDIF(PYTHONEXEC_NAME)
