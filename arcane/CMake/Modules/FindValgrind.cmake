#
# Find the 'valgrind' tool
#
# This module defines
# VALGRIND_EXEC_NAME, where to find valgrind binaries
 
FIND_PROGRAM(VALGRIND_EXEC_NAME valgrind)

SET( VALGRIND_FOUND "NO" )
IF(VALGRIND_EXEC_NAME)
  SET( VALGRIND_FOUND "YES" )
ENDIF(VALGRIND_EXEC_NAME)
