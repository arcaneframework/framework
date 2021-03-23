// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
%typemap(ctype) Arccore::String "const char*"
%typemap(ctype) const Arccore::String& "const char*"
%typemap(imtype) Arccore::String "string"
%typemap(imtype) const Arccore::String& "string"
%typemap(cstype) Arccore::String "string"
%typemap(cstype) const Arccore::String& "string"
%typemap(csin) Arccore::String "$csinput"
%typemap(csin) const Arccore::String& "$csinput"
%typemap(csout, excode=SWIGEXCODE) Arccore::String {
    string ret = $imcall;$excode
    return ret;
  }
%typemap(csout, excode=SWIGEXCODE) const Arccore::String& {
    string ret = $imcall;$excode
    return ret;
  }
%typemap(typecheck) Arccore::String = char *;
%typemap(typecheck) const Arccore::String& = char *;

%typemap(out) Arccore::String %{ $result = SWIG_csharp_string_callback($1.localstr()); %}
%typemap(out) const Arccore::String& %{ $result = SWIG_csharp_string_callback(($1)->localstr()); %}
%typemap(in) Arccore::String %{ Arccore::String $1_str{Arcane::fromCSharpCharPtr($input)}; $1 = $1_str; %}
%typemap(in) const Arccore::String& %{ Arccore::String $1_str{Arcane::fromCSharpCharPtr($input)}; $1 = &$1_str; %}
%typemap(directorin) Arccore::String %{ $input = SWIG_csharp_string_callback($1.localstr()); %}
%typemap(directorin) const Arccore::String& %{ $input = SWIG_csharp_string_callback($1.localstr()); %}
%typemap(directorout) Arccore::String %{ $result = Arccore::String{Arcane::fromCSharpCharPtr($input)}; %}
%typemap(directorout) const Arccore::String& %{
   static Arccore::String($1_str);
   $1_str = $input;
   $result = &$1_str;
 %}
%typemap(csdirectorin) const Arccore::String & "$iminput"
%typemap(csdirectorin) Arccore::String "$iminput"
%typemap(csdirectorout) const Arccore::String & "$cscall"
%typemap(csdirectorout) Arccore::String "$cscall"
