def extract_tracked_people(results):

    persons = []

    if results[0].boxes.id is None:
        return persons

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    for box, pid, conf in zip(boxes, ids, confs):

        x1, y1, x2, y2 = box

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        persons.append({
            "id": int(pid),
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "center": (cx, cy),
            "conf": float(conf)
        })

    return persons