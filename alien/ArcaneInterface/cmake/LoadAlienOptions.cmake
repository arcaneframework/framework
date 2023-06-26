# packages
# --------

# Chargement des options d'alien

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

createOption(COMMANDLINE ALIEN_IFPEN
	     NAME        ALIEN_COMPONENT_IFPEN
	     MESSAGE	 "Whether or not to compile IFPEN package"
	     DEFAULT	 ON)

createOption(COMMANDLINE ALIEN_CEA
	     NAME        ALIEN_COMPONENT_CEA
	     MESSAGE	 "Whether or not to compile CEA package"
	     DEFAULT	 ON)

createOption(COMMANDLINE ALIEN_EXTERNALS
	     NAME        ALIEN_COMPONENT_EXTERNALS
	     MESSAGE	 "Whether or not to compile EXTERNALS package"
	     DEFAULT	 ON)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
