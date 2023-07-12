#!/usr/bin/env perl

use warnings;
use strict;
#use XML::DOM;
use IO::Handle;
use IO::Pipe;
#use Template;
#use Template::Stash::XS;
use Getopt::Long;
#------------------------------------------------------------
# Main program
#------------------------------------------------------------

# Read arguments
my $template_filename    = shift or usage();
my $output_filename      = shift or usage();
my $param1         = shift or usage();
my $param2         = shift or usage();
my $param3         = shift or usage();

GetOptions("template-file:s" => \$template_filename,
           "output_filename:s" => \$output_filename,
           "param1:i"    => \$param1);
print "template file : $template_filename\n";
print "out file      : $output_filename\n";
my %replacements = ();
$replacements{PARAM1} = $param1;
$replacements{PARAM2} = $param2;
$replacements{PARAM3} = $param3;
open(IN, $template_filename) or die "Failed to open $template_filename";
open(OUT, ">$output_filename") or die "Failed to create $output_filename";
while(<IN>) 
{
  s/\[\%\s*(\w+)\s*\%\]/$replacements{$1}/g;
  print OUT;
}
close(IN);
close(OUT);
