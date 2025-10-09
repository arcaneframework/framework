// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
%typemap(ctype) Arccore::String "const char*"
%typemap(ctype) const Arccore::String& "const char*"
%typemap(imtype) Arccore::String "string"
%typemap(imtype) const Arccore::String& "string"
%typemap(cstype) Arccore::String "string"
%typemap(cstype) const Arccore::String& "string"
%typemap(csin) Arccore::String "$csinput"
%typemap(csin) const Arccore::String& "$csinput"
#if defined(SWIGCSHARP)
%typemap(csout, excode=SWIGEXCODE) Arccore::String {
    string ret = $imcall;$excode
    return ret;
  }
%typemap(csout, excode=SWIGEXCODE) const Arccore::String& {
    string ret = $imcall;$excode
    return ret;
  }
#endif
%typemap(typecheck) Arccore::String = char *;
%typemap(typecheck) const Arccore::String& = char *;


#if defined(SWIGCSHARP)
%typemap(out) Arccore::String %{ $result = SWIG_csharp_string_callback($1.localstr()); %}
%typemap(out) const Arccore::String& %{ $result = SWIG_csharp_string_callback(($1)->localstr()); %}
%typemap(in) Arccore::String %{ Arccore::String $1_str{Arcane::fromCSharpCharPtr($input)}; $1 = $1_str; %}
%typemap(in) const Arccore::String& %{ Arccore::String $1_str{Arcane::fromCSharpCharPtr($input)}; $1 = &$1_str; %}
%typemap(directorin) Arccore::String %{ $input = SWIG_csharp_string_callback($1.localstr()); %}
%typemap(directorin) const Arccore::String& %{ $input = SWIG_csharp_string_callback($1.localstr()); %}
%typemap(directorout) Arccore::String %{ $result = Arccore::String{Arcane::fromCSharpCharPtr($input)}; %}
#endif
#if defined(SWIGPYTHON)
%typemap(out) Arccore::String %{ $result = SWIG_Python_str_FromChar($1.localstr()); /* OUT */ %}
%typemap(out) const Arccore::String& %{ $result = SWIG_Python_str_FromChar(($1)->localstr()); /* OUT ref */ %}
%typemap(in) Arccore::String %{ $1 = SWIG_Python_str_AsChar($input); /* IN */ %}
//%typemap(in) const Arccore::String& %{ Arccore::String $1_str{PyUnicode_AsUTF8($input)}; $1 = &$1_str; /* IN ref */ %}
%typemap(directorin) Arccore::String %{ $input = SWIG_Python_str_FromChar($1.localstr()); /* DIRIN */ %}
%typemap(directorin) const Arccore::String& %{ $input = SWIG_Python_str_FromChar($1.localstr()); /* DIRIN ref */ %}
%typemap(directorout) Arccore::String %{ $result = Arccore::String{PyUnicode_AsUTF8($input)}; /* DIROUT */ %}
#endif

%typemap(directorout) const Arccore::String& %{
   static Arccore::String($1_str);
   $1_str = $input;
   $result = &$1_str;
 %}
%typemap(csdirectorin) const Arccore::String & "$iminput"
%typemap(csdirectorin) Arccore::String "$iminput"
%typemap(csdirectorout) const Arccore::String & "$cscall"
%typemap(csdirectorout) Arccore::String "$cscall"
