#! /usr/bin/env perl

$infile = shift(@ARGV);
$mtfile = shift(@ARGV);

open(MT, ">$mtfile");
binmode(MT, ":utf8");

@allsents = ();
ReadSent($infile, \@allsents);
$a = scalar(@allsents);
print "$a lines in $infile\n";
$begin = -1;
$end = -1;
$good = 0;
for($i = 0; $i < $a; $i++){
    $sent = $allsents[$i];
    if($sent =~ /^\# ::snt/){
	$begin = $i;
	$end = -1;
	$good = 0;
	for($j = $i+1; $j < $a; $j++){
	    $sent = $allsents[$j];
	    if($sent =~ /^\# ::node/){
		$sent =~ s/\t/ /g;
		@toks = split(/\s+/, $sent);
		$concept = $toks[3];
		if($concept =~ /polarity/ || $concept eq "-" ||
		    $concept =~ /have\-degree\-91/ || $concept =~ /imperative/){
		    $good = 1;
		}
	    }
	    elsif($sent =~ /^\s*$/){
		$end = $j;
		last;
	    }
	}
	if($good eq "1"){
	    for($m = $begin; $m < $end; $m++){
		$sent = $allsents[$m];
		$negsent = "";
		if($sent =~ /^\# ::tok/){
		    $sent =~ s/^\# ::tok //g;
		    @tokens = split(/\s+/, $sent);
		    $tc = scalar(@tokens);
		    $negsent = $sent;
		}
		if($sent =~ /^\# ::node/){
		    $sent =~ s/\t/ /g;
		    @toks = split(/\s+/, $sent);
		    $tc = scalar(@toks);
		    $concept = $toks[3];
		    if($concept eq "-" && $tc eq "5"){
			$align = $toks[4];
			@alns = split(/\-/, $align);
			$ac = scalar(@alns);
			if($ac != 2){
			    print "$m $align is no good...\n";
			    exit(1);
			}
			$idx = $alns[0];
			#print "negtoken: ", $tokens[$idx], "\n";
			if(substr($tokens[$idx],0,1) ne "n" && substr($tokens[$idx],0,1) ne "N" &&
			   substr($tokens[$idx],0,2) !~ /il/i &&
			   substr($tokens[$idx],0,2) !~ /im/i &&
			   substr($tokens[$idx],0,2) !~ /un/i &&
			   substr($tokens[$idx],0,2) !~ /in/i &&
			   $tokens[$idx] !~ /without/i){
			    $prev = $idx-1;
			    $nxt = $idx+1;
			    $nxt2 = $idx+2;
			    if($tokens[$nxt] eq "n't" || $tokens[$nxt] eq "without" ||
			       $tokens[$nxt] eq "not" || $tokens[$nxt] eq "Without"){
				$negidx = $nxt."-".$nxt2;
				$newsent = "";
				$newsent .= $toks[0];
				$newsent .= " ";
				$newsent .= $toks[1];
				$newsent .= "\t";
				$newsent .= $toks[2];
				$newsent .= "\t";
				$newsent .= $toks[3];
				$newsent .= "\t";
				$newsent .= $negidx;
			    }
			    else{
				$newsent = "";
				$newsent .= $toks[0];
				$newsent .= " ";
				$newsent .= $toks[1];
				$newsent .= "\t";
				$newsent .= $toks[2];
				$newsent .= "\t";
				$newsent .= $toks[3];
			    }
			}
			else{
			    $newsent = $allsents[$m];
			}
		    }
		    elsif($concept eq "have-degree-91" || $concept eq "imperative"){
			if($tc eq "5"){
			    $newsent = "";
			    $newsent .= $toks[0];
			    $newsent .= " ";
			    $newsent .= $toks[1];
			    $newsent .= "\t";
			    $newsent .= $toks[2];
			    $newsent .= "\t";
			    $newsent .= $toks[3];
			}
			else{
			    $newsent = $allsents[$m];
			}
		    }
		    else{
			$newsent = $allsents[$m];
		    }
                }
		else{
		    $newsent = $allsents[$m];
		}
		print MT $newsent, "\n";
	    }
	    print MT "\n";
	}
	else{
	    for($m = $begin; $m < $end; $m++){
		print MT $allsents[$m], "\n";
	    }
	    print MT "\n";
	}
    }
}

sub ReadSent{
    my $file = shift;
    my $sents = shift;

    open(FN, "$file");
    binmode(FN, ":utf8");

    while($line = <FN>){
	chomp($line);
	push @$sents, $line;
    }
    close(FN);
}
