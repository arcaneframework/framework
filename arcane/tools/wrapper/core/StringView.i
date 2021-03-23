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
%typemap(in) Arccore::StringView %{ Arccore::StringView $1_str{Arcane::fromCSharpCharPtrToStringView($input)}; $1 = $1_str; %}
%typemap(directorin) Arccore::StringView %{ $input = SWIG_csharp_string_callback($1.localstr()); %}
%typemap(csdirectorin) Arccore::StringView "$iminput"
