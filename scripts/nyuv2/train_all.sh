for f in *.sh; do
  if [ "$f" != "train_all.sh" ]; then 
    bash $f || break
  fi
done