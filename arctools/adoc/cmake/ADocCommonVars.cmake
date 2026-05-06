
set(DOXYGEN_HTML_EXTRA_STYLESHEET ${DOXYGEN_HTML_EXTRA_STYLESHEET}
  "${ADOC_SOURCE_DIR}/theme/doxygen-awesome-theme/doxygen-awesome.css"
  "${ADOC_SOURCE_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-sidebar-only.css"
  "${ADOC_SOURCE_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-sidebar-only-darkmode-toggle.css"
  "${ADOC_SOURCE_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-custom.css"
  "${ADOC_SOURCE_DIR}/theme/css/custom.css"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_HTML_EXTRA_FILES ${DOXYGEN_HTML_EXTRA_FILES}
  "${ADOC_SOURCE_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-darkmode-toggle.js"
  "${ADOC_SOURCE_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-fragment-copy-button.js"
  "${ADOC_SOURCE_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-paragraph-link.js"
  "${ADOC_SOURCE_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-interactive-toc.js"
  "${ADOC_SOURCE_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-tabs.js"
  "${ADOC_SOURCE_DIR}/theme/js/script-helper.js"
  "${ADOC_SOURCE_DIR}/theme/js/script-resize.js"
  "${ADOC_SOURCE_DIR}/theme/js/script-num-lines-code.js"
  "${ADOC_SOURCE_DIR}/theme/js/script-config-theme.js"
  "${ADOC_SOURCE_DIR}/theme/js/script-edit-config-theme.js"
  "${ADOC_SOURCE_DIR}/theme/js/script-apply-config-theme.js"
  "${ADOC_SOURCE_DIR}/theme/js/script-doxygen-version.js"
)

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_PROJECT_LOGO)
  set(DOXYGEN_PROJECT_LOGO
    "${ADOC_SOURCE_DIR}/theme/img/arcane_framework_small.webp"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_HTML_HEADER AND ${ADOC_LEGACY_THEME} STREQUAL "ON")
  set(DOXYGEN_HTML_HEADER
    "${ADOC_SOURCE_DIR}/theme/html/header_no_theme.html"
  )
else ()
  set(DOXYGEN_HTML_HEADER
    "${ADOC_BUILD_DIR}/theme/html/header.html"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_HTML_FOOTER)
  set(DOXYGEN_HTML_FOOTER
    "${ADOC_SOURCE_DIR}/theme/html/footer.html"
  )
endif ()

# ----------------------------------------------------------------------------

set(DOXYGEN_EXCLUDE_PATTERNS ${DOXYGEN_EXCLUDE_PATTERNS}
  "*/snippets/*"
)

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_CREATE_SUBDIRS)
  set(DOXYGEN_CREATE_SUBDIRS
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_OUTPUT_LANGUAGE)
  set(DOXYGEN_OUTPUT_LANGUAGE
    "French"
  )
endif ()

# ----------------------------------------------------------------------------

set(DOXYGEN_ABBREVIATE_BRIEF ${DOXYGEN_ABBREVIATE_BRIEF}
  "The $name class"
  "The $name widget"
  "The $name file"
  "is"
  "provides"
  "specifies"
  "contains"
  "represents"
  "a"
  "an"
  "the"
)


# ----------------------------------------------------------------------------

if (NOT DOXYGEN_FULL_PATH_NAMES)
  set(DOXYGEN_FULL_PATH_NAMES
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_TAB_SIZE)
  set(DOXYGEN_TAB_SIZE
    "2"
  )
endif ()

# ----------------------------------------------------------------------------

set(DOXYGEN_EXTENSION_MAPPING ${DOXYGEN_EXTENSION_MAPPING}
  "axl=xml"
  "arc=xml"
)

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_TOC_INCLUDE_HEADINGS)
  set(DOXYGEN_TOC_INCLUDE_HEADINGS
    "5"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_TYPEDEF_HIDES_STRUCT)
  set(DOXYGEN_TYPEDEF_HIDES_STRUCT
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_NUM_PROC_THREADS)
  set(DOXYGEN_NUM_PROC_THREADS
    "0"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_TIMESTAMP)
  set(DOXYGEN_TIMESTAMP
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_CASE_SENSE_NAMES)
  set(DOXYGEN_CASE_SENSE_NAMES
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_GENERATE_TESTLIST)
  set(DOXYGEN_GENERATE_TESTLIST
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_GENERATE_BUGLIST)
  set(DOXYGEN_GENERATE_BUGLIST
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

set(DOXYGEN_FILE_PATTERNS ${DOXYGEN_FILE_PATTERNS}
  "*.h"
  "*.H"
  "*.cc"
  "*.dox"
  "*.doc"
  "*.md"
)

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_SOURCE_BROWSER)
  set(DOXYGEN_SOURCE_BROWSER
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_REFERENCED_BY_RELATION)
  set(DOXYGEN_REFERENCED_BY_RELATION
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_REFERENCES_RELATION)
  set(DOXYGEN_REFERENCES_RELATION
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_HTML_COLORSTYLE)
  set(DOXYGEN_HTML_COLORSTYLE
    "TOGGLE"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_HTML_DYNAMIC_SECTIONS)
  set(DOXYGEN_HTML_DYNAMIC_SECTIONS
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_HTML_FORMULA_FORMAT)
  set(DOXYGEN_HTML_FORMULA_FORMAT
    "svg"
  )
endif ()

# ----------------------------------------------------------------------------

if (${ADOC_MATHJAX} STREQUAL "ON")
  set(DOXYGEN_USE_MATHJAX
    "YES"
  )
  set(DOXYGEN_MATHJAX_VERSION
    "MathJax_3"
  )
  set(DOXYGEN_MATHJAX_RELPATH
    "https://cdn.jsdelivr.net/npm/mathjax@3"
  )
else ()
  set(DOXYGEN_USE_MATHJAX
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_MACRO_EXPANSION)
  set(DOXYGEN_MACRO_EXPANSION
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_EXPAND_ONLY_PREDEF)
  set(DOXYGEN_EXPAND_ONLY_PREDEF
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_SKIP_FUNCTION_MACROS)
  set(DOXYGEN_SKIP_FUNCTION_MACROS
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_HAVE_DOT)
  set(DOXYGEN_HAVE_DOT
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_UML_LOOK)
  set(DOXYGEN_UML_LOOK
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_DOT_IMAGE_FORMAT)
  set(DOXYGEN_DOT_IMAGE_FORMAT
    "svg"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_INTERACTIVE_SVG)
  set(DOXYGEN_INTERACTIVE_SVG
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------
