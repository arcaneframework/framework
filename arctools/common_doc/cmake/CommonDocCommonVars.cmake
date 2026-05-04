
set(DOXYGEN_HTML_EXTRA_STYLESHEET
  "${COMMON_DOC_DIR}/theme/doxygen-awesome-theme/doxygen-awesome.css"
  "${COMMON_DOC_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-sidebar-only.css"
  "${COMMON_DOC_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-sidebar-only-darkmode-toggle.css"
  "${COMMON_DOC_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-custom.css"
  "${COMMON_DOC_DIR}/theme/css/custom.css"
)
set(DOXYGEN_HTML_EXTRA_STYLESHEET ${DOXYGEN_HTML_EXTRA_STYLESHEET}
  PARENT_SCOPE
)

set(DOXYGEN_HTML_EXTRA_FILES
  "${COMMON_DOC_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-darkmode-toggle.js"
  "${COMMON_DOC_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-fragment-copy-button.js"
  "${COMMON_DOC_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-paragraph-link.js"
  "${COMMON_DOC_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-interactive-toc.js"
  "${COMMON_DOC_DIR}/theme/doxygen-awesome-theme/doxygen-awesome-tabs.js"
  "${COMMON_DOC_DIR}/theme/js/script-helper.js"
  "${COMMON_DOC_DIR}/theme/js/script-resize.js"
  "${COMMON_DOC_DIR}/theme/js/script-num-lines-code.js"
  "${COMMON_DOC_DIR}/theme/js/script-config-theme.js"
  "${COMMON_DOC_DIR}/theme/js/script-edit-config-theme.js"
  "${COMMON_DOC_DIR}/theme/js/script-apply-config-theme.js"
  "${COMMON_DOC_DIR}/theme/js/script-doxygen-version.js"
)
set(DOXYGEN_HTML_EXTRA_FILES ${DOXYGEN_HTML_EXTRA_FILES}
  PARENT_SCOPE
)

if (${COMMON_DOC_LEGACY_THEME} STREQUAL "ON")
  set(DOXYGEN_HTML_HEADER
    "${COMMON_DOC_DIR}/theme/html/header_no_theme.html"
    PARENT_SCOPE
  )
else ()
  set(DOXYGEN_HTML_HEADER
    "${COMMON_DOC_DIR}/theme/html/header.html"
    PARENT_SCOPE
  )
endif ()


set(DOXYGEN_HTML_FOOTER
  "${COMMON_DOC_DIR}/theme/html/footer.html"
  PARENT_SCOPE
)


set(DOXYGEN_EXCLUDE_PATTERNS
  "*/snippets/*"
  PARENT_SCOPE
)

set(DOXYGEN_CREATE_SUBDIRS
  "YES"
  PARENT_SCOPE
)

set(DOXYGEN_OUTPUT_LANGUAGE
  "French"
)

set(DOXYGEN_ABBREVIATE_BRIEF
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


set(DOXYGEN_FULL_PATH_NAMES
  "NO"
  PARENT_SCOPE
)

set(DOXYGEN_TAB_SIZE
  "2"
  PARENT_SCOPE
)

set(DOXYGEN_EXTENSION_MAPPING
  "axl=xml"
  "arc=xml"
  PARENT_SCOPE
)

set(DOXYGEN_TOC_INCLUDE_HEADINGS
  "5"
  PARENT_SCOPE
)

set(DOXYGEN_TYPEDEF_HIDES_STRUCT
  "YES"
  PARENT_SCOPE
)

set(DOXYGEN_NUM_PROC_THREADS
  "0"
  PARENT_SCOPE
)
set(DOXYGEN_TIMESTAMP
  "YES"
  PARENT_SCOPE
)

set(DOXYGEN_CASE_SENSE_NAMES
  "YES"
  PARENT_SCOPE
)

set(DOXYGEN_GENERATE_TESTLIST
  "NO"
  PARENT_SCOPE
)
set(DOXYGEN_GENERATE_BUGLIST
  "NO"
  PARENT_SCOPE
)

set(DOXYGEN_FILE_PATTERNS
  "*.h"
  "*.H"
  "*.cc"
  "*.dox"
  "*.doc"
  "*.md"
  PARENT_SCOPE
)

set(DOXYGEN_SOURCE_BROWSER
  "YES"
  PARENT_SCOPE
)

set(DOXYGEN_REFERENCED_BY_RELATION
  "YES"
  PARENT_SCOPE
)
set(DOXYGEN_REFERENCES_RELATION
  "YES"
  PARENT_SCOPE
)

set(DOXYGEN_HTML_COLORSTYLE
  "TOGGLE"
  PARENT_SCOPE
)
set(DOXYGEN_HTML_DYNAMIC_SECTIONS
  "YES"
  PARENT_SCOPE
)
set(DOXYGEN_HTML_FORMULA_FORMAT
  "svg"
  PARENT_SCOPE
)

set(DOXYGEN_USE_MATHJAX
  "${COMMON_DOC_MATHJAX}"
  PARENT_SCOPE
)
if (${COMMON_DOC_MATHJAX} STREQUAL "YES")
  set(DOXYGEN_MATHJAX_VERSION
    "MathJax_3"
    PARENT_SCOPE
  )
  set(DOXYGEN_MATHJAX_RELPATH
    "https://cdn.jsdelivr.net/npm/mathjax@3"
    PARENT_SCOPE
  )
endif ()

set(DOXYGEN_MACRO_EXPANSION
  "YES"
  PARENT_SCOPE
)
set(DOXYGEN_EXPAND_ONLY_PREDEF
  "YES"
  PARENT_SCOPE
)

set(DOXYGEN_SKIP_FUNCTION_MACROS
  "NO"
  PARENT_SCOPE
)
set(DOXYGEN_HAVE_DOT
  "YES"
  PARENT_SCOPE
)
set(DOXYGEN_UML_LOOK
  "YES"
  PARENT_SCOPE
)

set(DOXYGEN_DOT_IMAGE_FORMAT
  "svg"
  PARENT_SCOPE
)

set(DOXYGEN_INTERACTIVE_SVG
  "YES"
  PARENT_SCOPE
)
