# Make a temporary copy
cp manuscript_refs-numbered.tex tmp_manuscript_refs-numbered_no-sc.tex

echo "Replacing things that don't work well with pandoc..."

# Replace SC figures with regular figures
echo "\tSidecaption figures..."
sed -i '' 's|SCfigure\}\[.*\]|figure}|g' tmp_manuscript_refs-numbered_no-sc.tex
sed -i '' 's|SCfigure|figure|g' tmp_manuscript_refs-numbered_no-sc.tex

# Replace hspace in Figure 1
sed -i '' 's|\hspace{0.5cm}||g' tmp_manuscript_refs-numbered_no-sc.tex

# Replace \degree
echo "\tDegree symbol..."
sed -i '' 's|\$\\degree\$|°|g' tmp_manuscript_refs-numbered_no-sc.tex

# Replace emergency stretch
echo "\tEmergency stretch"
sed -i '' 's|\\emergencystretch=1em \% prevent overfull \\hbox for 1.5pt||g' tmp_manuscript_refs-numbered_no-sc.tex

# Convert with pandoc
pandoc tmp_manuscript_refs-numbered_no-sc.tex -f latex -t docx -o docx/manuscript_refs-numbered.docx --reference-doc resources/apa6_man.docx --csl=resources/nature.csl --metadata=notes-after-punctuation:false --metadata=link-citations:true --filter pandoc-crossref --citeproc 

# Remove files
rm tmp_manuscript_refs-numbered_no-sc*

# Print todos
echo "Things that need to be done manually:"
echo "  [ ] Create title page with abstract"
echo "  [ ] Center tables and figures"
echo "  [ ] Fix table captions (Captions below tables, include label)"
echo "  [ ] Number equations and replace equation labels with numbers"
echo "  [ ] Split Supplementary Information from main file"
echo "  [ ] Re-make Supplementary Table 2"
echo "  [ ] Re-number Supplementary Information References"

