import React from 'react';
import {useHistory} from '@docusaurus/router';

export default function Home(): React.ReactElement {
  const history = useHistory();
  React.useEffect(() => {
    history.push('/intro');
  }, [history]);
  return null;
}
