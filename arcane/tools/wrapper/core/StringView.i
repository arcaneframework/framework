// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*
 * Wrapping de la classe 'Arccore::StringView'.
 *
 * La classe C# correspondante sera 'string'. La classe 'StringView' ne
 * doit être utilisée que comme paramètre d'entrée et le passage se
 * fait uniquement par valeur.
 */
//
%typemap(ctype) Arccore::StringView "const char*"
%typemap(imtype) Arccore::StringView "string"
%typemap(cstype) Arccore::StringView "string"
%typemap(csin) Arccore::StringView "$csinput"
%typemap(typecheck) Arccore::StringView = char *;
#if defined(SWIGCSHARP)
%typemap(in) Arccore::StringView %{ Arccore::StringView $1_str{Arcane::fromCSharpCharPtrToStringView($input)}; $1 = $1_str; /* IN */%}
%typemap(directorin) Arccore::StringView %{ $input = SWIG_csharp_string_callback($1.localstr()); %}
#endif
#if defined(SWIGPYTHON)
// TODO: Utiliser PyUnicode_AsUTF8AndSize
%typemap(in) Arccore::StringView %{ $1 = PyUnicode_AsUTF8($input); /* IN */%}
%typemap(directorin) Arccore::StringView %{ $input = SWIG_Python_str_FromChar($1.localstr()); %}
#endif
%typemap(csdirectorin) Arccore::StringView "$iminput"
